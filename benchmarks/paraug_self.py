"""paraug self-benchmark.

Measures throughput (images/second) and median latency (ms/batch) of paraug
across a matrix of image sizes × batch sizes × devices for a representative
augmentation pipeline.

This is intentionally a **self-benchmark only** — it does not compare
paraug to other augmentation libraries. Cross-library comparison requires
careful methodology (matching primitive semantics across implementations,
warming up each lib's import / cache, etc.) and is deferred to v0.2.

Methodology:
  - Each (image_size, batch, device) combination runs:
      1. Warmup: 5 iterations (excluded from timing) to amortise the first
         `grid_sample` JIT specialisation and CUDA kernel cache.
      2. Measurement: 20 iterations, take the median of per-iteration time
         to dodge outlier-sensitive mean.
  - CUDA timing uses `torch.cuda.synchronize()` around each iteration so
    we measure end-to-end batch latency, not async kernel launch time.
  - Per-item seed advances by iteration index so different iterations
    sample different aug params (more realistic than fixed seed which
    would let the GPU reuse cached random buffers).
  - Pipeline mirrors a realistic training-time config:
      affine + tps (geometric, grid_sample class)
      gamma + gaussian_blur (photometric, elementwise + conv class)
      gaussian_noise (photometric, elementwise)

Run:
    python benchmarks/paraug_self.py
    python benchmarks/paraug_self.py --device cpu        # CPU-only sweep
    python benchmarks/paraug_self.py --device cuda       # CUDA-only sweep
    python benchmarks/paraug_self.py --sizes 256 512     # custom image sizes
    python benchmarks/paraug_self.py --batches 1 8 32    # custom batch sizes
    python benchmarks/paraug_self.py --json out.json     # also write JSON
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

# Local-dev shim: if paraug isn't installed (e.g. running from repo root),
# add src/ to path.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from paraug import AugPipeline


DEFAULT_CFG = {
    "geometric": {
        "affine": {"p": 1.0, "rot_deg": 15.0, "scale_range": (0.9, 1.1)},
        "tps":    {"p": 1.0, "max_disp": 12.0, "n_ctrl": 5},
    },
    "photometric": {
        "gamma":          {"p": 1.0, "gamma_range": (0.8, 1.2)},
        "gaussian_blur":  {"p": 1.0, "sigma_range": (0.5, 2.0)},
        "gaussian_noise": {"p": 1.0, "sigma_range": (0.005, 0.04)},
    },
}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_one(aug, img, mask, device, n_warmup, n_iter, seed_base):
    """Returns dict with median_ms, p5_ms, p95_ms, throughput_imgs_per_s, n_iter."""
    # Warmup
    for i in range(n_warmup):
        out_img, out_mask = aug(img, mask=mask, seed_base=seed_base + i,
                                 epoch=0, step=0)
    _sync(device)

    times = []
    for i in range(n_iter):
        _sync(device)
        t0 = time.perf_counter()
        out_img, out_mask = aug(img, mask=mask,
                                 seed_base=seed_base + n_warmup + i,
                                 epoch=0, step=0)
        _sync(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    times.sort()
    median_ms = statistics.median(times)
    p5_ms = times[max(0, int(0.05 * len(times)))]
    p95_ms = times[min(len(times) - 1, int(0.95 * len(times)))]
    B = img.shape[0]
    throughput = (B * 1000.0) / median_ms if median_ms > 0 else float("inf")
    return {
        "median_ms": round(median_ms, 3),
        "p5_ms": round(p5_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "throughput_imgs_per_s": round(throughput, 1),
        "n_iter": n_iter,
    }


def device_label(device):
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = "cuda"
        return f"cuda ({name})"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024],
                    help="image sizes (square, default: 256 512 1024)")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32],
                    help="batch sizes (default: 1 8 32)")
    p.add_argument("--device", choices=["cpu", "cuda", "both"], default="both",
                    help="device to benchmark (default: both)")
    p.add_argument("--warmup", type=int, default=5,
                    help="warmup iterations (default: 5)")
    p.add_argument("--iter", type=int, default=20,
                    help="measurement iterations (default: 20)")
    p.add_argument("--seed", type=int, default=42,
                    help="base seed (default: 42)")
    p.add_argument("--json", type=Path, default=None,
                    help="write raw results to this JSON file as well")
    args = p.parse_args()

    devices = []
    if args.device in ("cpu", "both"):
        devices.append(torch.device("cpu"))
    if args.device in ("cuda", "both") and torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if not devices:
        print("No device selected (or CUDA requested but not available)")
        return 1

    print(f"paraug self-benchmark")
    print(f"  pipeline:  {sorted(list(DEFAULT_CFG['geometric']) + list(DEFAULT_CFG['photometric']))}")
    print(f"  warmup:    {args.warmup}")
    print(f"  iter:      {args.iter}")
    print(f"  sizes:     {args.sizes}")
    print(f"  batches:   {args.batches}")
    print(f"  devices:   {[device_label(d) for d in devices]}")
    print()

    aug = AugPipeline(DEFAULT_CFG)

    results = []
    for device in devices:
        for size in args.sizes:
            for batch in args.batches:
                g = torch.Generator(device="cpu").manual_seed(args.seed)
                img = torch.rand(batch, 3, size, size, generator=g).to(device)
                mask = torch.ones(batch, 1, size, size, device=device)
                stats = time_one(aug, img, mask, device,
                                  args.warmup, args.iter, args.seed)
                row = {
                    "device": device_label(device),
                    "image_size": f"{size}x{size}",
                    "batch": batch,
                    **stats,
                }
                results.append(row)
                print(f"  {row['device']:30s}  "
                      f"{row['image_size']:>9s}  bs={row['batch']:>2d}  "
                      f"median={row['median_ms']:7.2f} ms  "
                      f"(p5={row['p5_ms']:6.2f}, p95={row['p95_ms']:6.2f})  "
                      f"throughput={row['throughput_imgs_per_s']:7.1f} imgs/s")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({
                "config": DEFAULT_CFG,
                "warmup": args.warmup,
                "iter": args.iter,
                "results": results,
            }, f, indent=2)
        print(f"\nwrote raw results → {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
