import re
import sys
import types
import time
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import ast
import os

from process_cache import search_for_file, config, get_best_config


os.environ.setdefault("TRITON_CACHE_DIR", "./triton_cache")
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")

def has_xpu(): return hasattr(torch, "xpu") and torch.xpu.is_available()
def sync():
    if has_xpu(): torch.xpu.synchronize()

def install_inductor_shims():
    """
    Provide minimal stand-ins for symbols some Inductor-generated Triton code imports.
    Works across PyTorch versions where these aren't exported.
    """
    mod_name = "torch._inductor.runtime.triton_heuristics"
    m = sys.modules.get(mod_name)
    if m is None:
        m = types.ModuleType(mod_name)
        sys.modules[mod_name] = m

    # grid(...) -> return a tuple (1D/2D/3D)
    if not hasattr(m, "grid"):
        def grid(*sizes, **kwargs):
            if sizes and callable(sizes[0]):  # e.g., grid(lambda META: ...)
                return (1,)
            if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
                return tuple(sizes[0])
            return tuple(sizes)
        m.grid = grid

    # Minimal FixedGrid that acts like a tuple of dims
    if not hasattr(m, "FixedGrid"):
        class FixedGrid:
            def __init__(self, *sizes):
                # allow FixedGrid((x,y,z)) or FixedGrid(x,y,z)
                if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
                    self.sizes = tuple(sizes[0])
                else:
                    self.sizes = tuple(sizes)
            # Behave tuple-like
            def __iter__(self):
                return iter(self.sizes)
            def __len__(self):
                return len(self.sizes)
            def __getitem__(self, i):
                return self.sizes[i]
            def __repr__(self):
                return f"FixedGrid{self.sizes}"
            # Some code may call it (rare); just return the tuple
            def __call__(self, *_, **__):
                return self.sizes
        m.FixedGrid = FixedGrid

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
    _install_triton_grid_shim()
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

# def infer_xnumel_for_kernel(src: str, kernel_name: str) -> Optional[int]:
#     """
#     Look for 'def kernel(...):' followed by 'xnumel = <int>' inside body.
#     Works well for Inductor-generated 1D kernels.
#     """
#     # Find the function block for this kernel
#     # crude but effective: grab from 'def name' to next 'def ' or EOF
#     pat = rf"@triton\.jit\s*def\s+{re.escape(kernel_name)}\s*\(.*?\):(.+?)(?=\n@triton\.jit|\ndef\s|\Z)"
#     m = re.search(pat, src, flags=re.DOTALL)
#     if not m:
#         return None
#     body = m.group(1)
#     m2 = re.search(r"\bxnumel\s*=\s*(\d+)", body)
#     return int(m2.group(1)) if m2 else None

def infer_xnumel_for_kernel(src: str, kernel_name: str) -> Tuple[Optional[int], Optional[List[str]]]:
    """
    Find the Triton kernel by name, capture its parameter list (inside the
    parentheses) and its body, then:
      - parse and return the parameter names
      - look for 'xnumel = <int>' inside the body and return that int

    Returns: (xnumel, param_names)
    """
    # Capture both the params and the body with named groups
    pat = rf"@triton\.jit\s*def\s+{re.escape(kernel_name)}\s*\((?P<params>.*?)\)\s*:(?P<body>.+?)(?=\n@triton\.jit|\ndef\s|\Z)"
    m = re.search(pat, src, flags=re.DOTALL)
    if not m:
        return None, None

    params_src = m.group("params").strip()
    body = m.group("body")

    # Robustly parse parameter *names* via AST (handles annotations/defaults)
    param_names: Optional[List[str]] = None
    try:
        dummy = f"def __tmp__({params_src}):\n    pass\n"
        mod = ast.parse(dummy)
        f = mod.body[0]  # type: ignore[index]
        # Collect positionals; add others if you care (vararg/kwonly/kwarg)
        posonly = getattr(f.args, "posonlyargs", [])
        args = f.args.args
        param_names = [a.arg for a in (*posonly, *args)]
        # Optionally include vararg/kwonly/kwarg:
        # if f.args.vararg: param_names.append("*" + f.args.vararg.arg)
        # param_names += [a.arg for a in f.args.kwonlyargs]
        # if f.args.kwarg: param_names.append("**" + f.args.kwarg.arg)
    except Exception:
        # Fallback: naive split (good enough for simple signatures)
        if params_src:
            param_names = [p.split(":", 1)[0].split("=", 1)[0].strip()
                           for p in params_src.split(",") if p.strip()]

    # Extract xnumel literal from the body, if present
    m2 = re.search(r"\bxnumel\s*=\s*(\d+)", body)
    xnumel = int(m2.group(1)) if m2 else None

    return xnumel, param_names

# ---------------------------
# 3) Build dummy args
# ---------------------------
def make_dummy_args_for_kernel(param_names: List[str], xnumel: int, device: torch.device):
    """
    Heuristic arg builder for 1D Inductor Triton kernels.
    Assumes signature like:
        (in_ptr0, in_ptr1, ..., xnumel, XBLOCK: tl.constexpr)
    We allocate float32 tensors for pointer-like args and pass xnumel as int.
    """
    args = []
    kw = {}

    # Count how many "pointer-like" args come before 'xnumel'
    # For Inductor kernels, names usually end with _ptr0, in_ptr0, out_ptr0, in_out_ptr0, etc.
    ptr_params = []
    saw_xnumel = False
    for name in param_names:
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
    print(f"  args: {[type(a) for a in args]}")

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
                    ms = bench(run, warmup=1, iters=2)
                except Exception as e:
                    print(f"  [ERROR] {kernel_name} XBLOCK={xb} warps={w} stages={s} - {e}")
                    errors += 1
                    continue
                if ms < best["lat_ms"]:
                    best = {"lat_ms": ms, "cfg": {"XBLOCK": xb, "num_warps": w, "num_stages": s, "grid": g}}

    return {"kernel": kernel_name, "xnumel": xnumel, "attempts": attempts, "errors": errors, "best": best}

import triton
import triton.language as tl

def xpu_do_bench(fn, quantiles=(0.5,)):
    """
    fn: a zero-arg callable that launches the kernel once.
    quantiles: ignored except we return the median (0.5).
    Returns: median latency in *milliseconds*.
    """
    # warmup
    for _ in range(25):
        fn(); sync()
    # measure
    iters = 100
    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(); sync()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1e3)
    times_ms.sort()
    # return median by default
    time_output = times_ms[len(times_ms)//2]
    return time_output

def tune_kernel(kernel_fn, kernel_name: str, xnumel: int, param_names:List[str], device: torch.device) -> Dict:
    """
    Wrap `kernel_fn` with Triton's autotuner at runtime, trigger autotune once
    (for this problem size), then benchmark the cached best config.

    Assumptions:
      - `kernel_fn` is a Triton JIT kernel that accepts a meta-parameter `XBLOCK`
        and launches like: kernel_fn[grid](*args, XBLOCK=..., num_warps=..., num_stages=...)
      - `grid_1d(xnumel, XBLOCK)` exists and returns (cdiv(xnumel, XBLOCK),)
      - `make_dummy_args_for_kernel` and `bench` are available
    """
    # Prepare dummy args
    # print(f"  params: {param_names}")
    args, _ = make_dummy_args_for_kernel(param_names, xnumel, device)
    # print(f"  args: {[type(a) for a in args]}")

    # Candidate search space
    # XBLOCKS = [16, 32, 64, 128, 256]
    # WARPS   = [1, 2, 4, 8]
    # STAGES  = [1, 2, 3]
    XBLOCKS = [16, 32,]
    WARPS   = [1, 2]
    STAGES  = [1, 2,]

    # Build Triton autotune configs
    CONFIGS = [
        triton.Config({'XBLOCK': xb}, num_warps=w, num_stages=s)
        for xb in XBLOCKS for w in WARPS for s in STAGES
    ]

    # Autotune key: if your kernel doesn't take size params (e.g., xnumel) as an arg,
    # we use a global key ([]). If your kernel *does* take xnumel (or other dims)
    # as a kernel arg, replace [] with ['xnumel', ...] so caching is per-shape.
    KEY = []

    # Wrap the JIT kernel with autotune.
    # In Triton, the autotune decorator typically wraps the @triton.jit function (outermost).
    # Here we apply it dynamically to the already-jitted function object.
    autotuned_kernel = triton.autotune(configs=CONFIGS, key=KEY, do_bench=None, cache_results = True)(kernel_fn)

    # Grid lambda that lets autotune inject each config's XBLOCK
    grid = (lambda meta: grid_1d(xnumel, meta['XBLOCK']))

    # --- Trigger autotune once (runs all viable configs & caches the winner) ---
    try:
        autotuned_kernel[grid](*args)
    except Exception:
        # If some configs are invalid on this device, autotuner will skip them internally.
        # A failure here usually means signature mismatch; fall back to first valid call.
        pass

    # --- Benchmark the tuned kernel (uses the cached best config) ---
    lat_ms = bench(lambda: autotuned_kernel[grid](*args), warmup=3, iters=10)
    
    config_search = config()
    config_search.name = "autotune.json"
    res = search_for_file(config_search)
    if res:
        print(f"Found config file: {res}")
    else:
        print("No config file found.")
        
    best_cfg = get_best_config(res) if res else None
    print(f"Best config from cache: {best_cfg}")
    _placeholder_xb = best_cfg.get("block_size") if best_cfg else None

    # We don't rely on private Triton internals to read back the exact picked config.
    # If you *must* see it, set TRITON_LOG_AUTOTUNING=1 in the environment before running.
    result = {
        "kernel": kernel_name,
        "xnumel": xnumel,
        "attempts": len(CONFIGS),      # autotune tries each viable config exactly once for this key
        "errors": None,                # not exposed cleanly by public API
        "best": {
            "lat_ms": lat_ms,
            "cfg": "selected by triton.autotune (see TRITON_LOG_AUTOTUNING=1 for details)",
            "grid": triton.cdiv(xnumel, _placeholder_xb),  # for reference; XBLOCK is from the winning config at runtime
            "num_warps": best_cfg.get("num_warps"),
            "num_stages": best_cfg.get("num_stages"),
            "XBLOCK": _placeholder_xb
        },
    }
    return result


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
            xnumel, param_names = infer_xnumel_for_kernel(src, name)
            if not xnumel:
                print(f"[SKIP] Could not infer xnumel for {name}")
                continue
            print(f"[TUNE] {name}  (xnumel={xnumel})")
            res = tune_kernel(obj, name, xnumel, param_names, torch.device(device))
            print("   best:", res["best"])
            results.append(res)

    return results

if __name__ == "__main__":
    # Example usage
    # python tune_one_file.py /path/to/that_triton_file.py
    path = sys.argv[1]
    install_inductor_shims()
    out = autotune_file(path)
    print(out)
