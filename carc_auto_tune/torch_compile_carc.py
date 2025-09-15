#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate verbose TorchInductor/Triton autotune logs on NVIDIA GPUs.

What this does:
- Sets env vars (before importing torch) to maximize logging.
- Flips Inductor config flags for "max_autotune".
- Builds two CUDA workloads:
  1) Conv2d block (common for vision)
  2) GEMM-heavy MLP (Linear layers)
- Runs with torch.compile so Inductor/Triton kicks in.
- Writes BOTH stdout and stderr to a log file (and still prints to console).

Tip:
- If you don't see much autotune chatter, try a bigger input size or more warmup/iters.
- Different PyTorch versions print slightly different strings; still fine for the parser.
"""

import os
import sys
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

# ---- Set logging env BEFORE importing torch ----
# Make Dynamo + Inductor chatty
os.environ.setdefault("TORCH_LOGS", "+dynamo,+inductor")
# Optional: more aggressive kernel/tuning paths in many builds
os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "1")
# Where some inductor logs/caches may land (not guaranteed by all builds)
os.environ.setdefault("TORCHINDUCTOR_LOG_DIR", "./logs_inductor")
# Triton cache dir (kernels, sometimes useful for debugging)
os.environ.setdefault("TRITON_CACHE_DIR", "./triton_cache")

# Some builds emit more details with DEBUG; harmless if ignored
os.environ.setdefault("PYTORCH_JIT_LOG_LEVEL", ">>DEBUG")

# Now it’s safe to import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Extra inductor flags via Python (safe to ignore if not present in your build)
try:
    from torch._inductor import config as inductor_cfg
    # Maximize autotune in Inductor (names can vary across versions; these are common)
    if hasattr(inductor_cfg, "max_autotune"):
        inductor_cfg.max_autotune = True
    if hasattr(inductor_cfg, "max_autotune_gemm"):
        inductor_cfg.max_autotune_gemm = True
    if hasattr(inductor_cfg, "max_autotune_points"):
        inductor_cfg.max_autotune_points = True
    # Sometimes helps print more per-candidate timing info
    if hasattr(inductor_cfg, "debug"):
        inductor_cfg.debug = True
except Exception:
    pass  # Not fatal if the internal config structure changed


class ConvBlock(nn.Module):
    def __init__(self, in_ch=64, out_ch=128, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GemmMLP(nn.Module):
    def __init__(self, d_in=4096, d_hidden=8192, d_out=4096):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        # GEMM-heavy path
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


def run_conv_block(device="cuda", iters=10, warmup=5):
    print("\n[run] Conv2d block")
    m = ConvBlock().to(device)
    x = torch.randn(32, 64, 112, 112, device=device)  # NCHW
    m = torch.compile(m, backend="inductor", mode="max-autotune")
    # Warmup
    for _ in range(warmup):
        y = m(x)
        torch.cuda.synchronize()
    # Timed iters
    for i in range(iters):
        y = m(x)
        torch.cuda.synchronize()
        if i == 0:
            print("[info] first conv iter done (expect autotune around here)")


def run_gemm_mlp(device="cuda", iters=10, warmup=5):
    print("\n[run] GEMM MLP")
    m = GemmMLP().to(device)
    x = torch.randn(256, 4096, device=device)  # (batch, features)
    m = torch.compile(m, backend="inductor", mode="max-autotune")
    # Warmup
    for _ in range(warmup):
        y = m(x)
        torch.cuda.synchronize()
    # Timed iters
    for i in range(iters):
        y = m(x)
        torch.cuda.synchronize()
        if i == 0:
            print("[info] first gemm iter done (expect autotune around here)")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. This script is intended for NVIDIA GPUs.", file=sys.stderr)
        sys.exit(1)

    # Create a unique log file name
    os.makedirs("logs", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logfile = os.path.join("logs", f"torch_compile_autotune-{stamp}.log")

    # Tee stdout/stderr to file and console
    class Tee:
        def __init__(self, stream, log_handle):
            self.stream = stream
            self.log_handle = log_handle
        def write(self, data):
            self.stream.write(data)
            self.log_handle.write(data)
        def flush(self):
            self.stream.flush()
            self.log_handle.flush()

    with open(logfile, "w", encoding="utf-8") as fh:
        tee_out = Tee(sys.stdout, fh)
        tee_err = Tee(sys.stderr, fh)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"[info] Writing combined logs to {logfile}")
            print(f"[info] TORCH_LOGS={os.environ.get('TORCH_LOGS')}")
            print(f"[info] TORCHINDUCTOR_MAX_AUTOTUNE={os.environ.get('TORCHINDUCTOR_MAX_AUTOTUNE')}")
            print(f"[info] TRITON_CACHE_DIR={os.environ.get('TRITON_CACHE_DIR')}")
            print(f"[info] PyTorch version: {torch.__version__}")
            print(f"[info] Device name: {torch.cuda.get_device_name(0)}")

            # Small CUDA op to initialize context (so you see CUDA+driver info early)
            torch.cuda.synchronize()

            # Run workloads
            run_conv_block()
            # run_gemm_mlp()

            print("\n[done] Finished runs. Search this file for keywords like:")
            print("       'autotune', 'max_autotune', 'candidate', 'selected config', 'best config', 'time', 'latency'.")

    print(f"\nLog saved to: {logfile}")


if __name__ == "__main__":
    main()
