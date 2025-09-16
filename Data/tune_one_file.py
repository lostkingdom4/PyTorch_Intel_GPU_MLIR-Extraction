import re
import sys
import types
import time
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch

# ---------------------------
# 0) Make 'grid' import safe
# ---------------------------
def _install_triton_grid_shim():
    """
    Inductor code often does:
      from torch._inductor.runtime.triton_heuristics import grid
    Some PyTorch builds don't export it. Provide a tiny shim.
    """
    mod_name = 'torch._inductor.runtime.triton_heuristics'
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        def grid(*sizes, **kwargs):
            if sizes and callable(sizes[0]):
                # if someone passed a lambda META: ... ; return harmless default
                return (1,)
            if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
                return tuple(sizes[0])
            return tuple(sizes)
        m.grid = grid
        sys.modules[mod_name] = m

_install_triton_grid_shim()

# ---------------------------
# 1) Dynamic import helpers
# ---------------------------
def import_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# ---------------------------
# 2) Source parsing: xnumel
# ---------------------------
def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def infer_xnumel_for_kernel(src: str, kernel_name: str) -> Optional[int]:
    """
    Look for 'def kernel(...):' followed by 'xnumel = <int>' inside body.
    Works well for Inductor-generated 1D kernels.
    """
    # Find the function block for this kernel
    # crude but effective: grab from 'def name' to next 'def ' or EOF
    pat = rf"@triton\.jit\s*def\s+{re.escape(kernel_name)}\s*\(.*?\):(.+?)(?=\n@triton\.jit|\ndef\s|\Z)"
    m = re.search(pat, src, flags=re.DOTALL)
    if not m:
        return None
    body = m.group(1)
    m2 = re.search(r"\bxnumel\s*=\s*(\d+)", body)
    return int(m2.group(1)) if m2 else None

# ---------------------------
# 3) Build dummy args
# ---------------------------
def make_dummy_args_for_kernel(fn, xnumel: int, device: torch.device):
    """
    Heuristic arg builder for 1D Inductor Triton kernels.
    Assumes signature like:
        (in_ptr0, in_ptr1, ..., xnumel, XBLOCK: tl.constexpr)
    We allocate float32 tensors for pointer-like args and pass xnumel as int.
    """
    import inspect
    sig = inspect.signature(fn)
    args = []
    kw = {}

    # Count how many "pointer-like" args come before 'xnumel'
    # For Inductor kernels, names usually end with _ptr0, in_ptr0, out_ptr0, in_out_ptr0, etc.
    ptr_params = []
    saw_xnumel = False
    for name, p in sig.parameters.items():
        if name == "xnumel":
            saw_xnumel = True
            break
        ptr_params.append(name)

    # allocate flat buffers length >= xnumel
    # (float32 is fine because loads/stores in your example use default dtype)
    buffers = []
    for _ in ptr_params:
        buffers.append(torch.empty(xnumel, dtype=torch.float32, device=device))

    # positional: all buffers, then xnumel
    args.extend(buffers)
    args.append(xnumel)

    # meta-params (constexpr) will be passed as **kw later (XBLOCK, num_warps, num_stages)
    return args, kw

# ---------------------------
# 4) Timing
# ---------------------------
def device_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()

def bench(fn, warmup=5, iters=20):
    times = []
    # warmup
    for _ in range(warmup):
        fn(); device_sync()
    for _ in range(iters):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(); fn(); end.record(); end.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter(); fn(); device_sync()
            times.append((time.perf_counter() - t0)*1000.0)
    times.sort()
    return times[len(times)//2]

# ---------------------------
# 5) Grid helpers
# ---------------------------
def ceil_div(a, b):
    return (a + b - 1) // b

def grid_1d(xnumel: int, XBLOCK: int):
    return (ceil_div(xnumel, XBLOCK),)

# ---------------------------
# 6) Tuning one kernel
# ---------------------------
def tune_kernel(kernel_fn, kernel_name: str, xnumel: int, device: torch.device) -> Dict:
    args, _ = make_dummy_args_for_kernel(kernel_fn, xnumel, device)

    # param space – small but meaningful for 1D pointwise kernels
    XBLOCKS   = [16, 32, 64, 128, 256]
    WARPS     = [1, 2, 4, 8]
    STAGES    = [1, 2, 3]

    best = {"lat_ms": float("inf"), "cfg": None}
    attempts = 0
    errors = 0

    for xb in XBLOCKS:
        g = grid_1d(xnumel, xb)
        for w in WARPS:
            for s in STAGES:
                attempts += 1
                try:
                    # Build a no-arg callable that launches with this config
                    def run():
                        kernel_fn[g](*args, XBLOCK=xb, num_warps=w, num_stages=s)
                    ms = bench(run, warmup=3, iters=10)
                except Exception:
                    errors += 1
                    continue
                if ms < best["lat_ms"]:
                    best = {"lat_ms": ms, "cfg": {"XBLOCK": xb, "num_warps": w, "num_stages": s, "grid": g}}

    return {"kernel": kernel_name, "xnumel": xnumel, "attempts": attempts, "errors": errors, "best": best}

# ---------------------------
# 7) Walk a module and tune
# ---------------------------
def autotune_file(py_path: str, device: str = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mod_name = f"tune_mod_{Path(py_path).stem}"
    src = read_text(Path(py_path))
    m = import_from_path(mod_name, py_path)

    results = []
    for name, obj in vars(m).items():
        # Inductor kernels are jit-compiled callables (decorated with @triton.jit)
        if callable(obj) and name.startswith("triton_poi_"):
            xnumel = infer_xnumel_for_kernel(src, name)
            if not xnumel:
                print(f"[SKIP] Could not infer xnumel for {name}")
                continue
            print(f"[TUNE] {name}  (xnumel={xnumel})")
            res = tune_kernel(obj, name, xnumel, torch.device(device))
            print("   best:", res["best"])
            results.append(res)

    return results

if __name__ == "__main__":
    # Example usage
    # python tune_one_file.py /path/to/that_triton_file.py
    path = sys.argv[1]
    out = autotune_file(path)
    print(out)
