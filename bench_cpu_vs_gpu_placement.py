"""Bench paraug per-sample CPU vs batch GPU placement.

The first paraug user pothole: instinct (from torchvision) is to put aug
inside Dataset.__getitem__, which loses GPU acceleration and ping-pongs
between worker processes. Right place is the train loop on a GPU batch.

This bench quantifies the gap for the README so users don't have to find
out the hard way.

Run: python bench_cpu_vs_gpu_placement.py
"""
import gc
import time

import torch

from paraug import AugPipeline


CFG = {
    "geometric": {
        "affine":  {"p": 1.0, "rot_deg": 15.0, "scale_range": (0.9, 1.1)},
        "perspective": {"p": 0.5, "max_disp_frac": 0.08},
    },
    "photometric": {
        "color_jitter":  {"p": 0.5, "brightness": 0.2, "contrast": 0.2},
        "gamma":         {"p": 0.5, "gamma_range": (0.8, 1.2)},
        "gaussian_blur": {"p": 0.3, "sigma_range": (0.5, 1.5)},
    },
}

B = 32
H = W = 224
N_ITER = 50


def bench_per_sample_cpu(aug, n_iter=N_ITER):
    """Mimic torchvision-style Dataset placement: per-sample CPU augmentation
    then stack to batch and ship to GPU."""
    times = []
    for i in range(n_iter):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        outs = []
        for j in range(B):
            x = torch.rand(1, 3, H, W)              # one CPU sample
            x_aug, _ = aug(x, seed_base=i * B + j)
            outs.append(x_aug)
        batch_cpu = torch.cat(outs, dim=0)
        batch_gpu = batch_cpu.to("cuda", non_blocking=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.perf_counter() - t0)
    return times


def bench_batch_gpu(aug, n_iter=N_ITER):
    """Right placement: load batch, ship to GPU, aug the whole batch on GPU."""
    times = []
    for i in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch_cpu = torch.rand(B, 3, H, W)          # whole batch on CPU
        batch_gpu = batch_cpu.to("cuda", non_blocking=True)
        batch_aug, _ = aug(batch_gpu, seed_base=i)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    import paraug; print(f"paraug: {paraug.__version__}")
    print(f"bs={B}, canvas={H}x{W}, n_iter={N_ITER}")

    aug = AugPipeline(CFG)

    # Warmup
    for _ in range(3):
        x = torch.rand(B, 3, H, W, device="cuda")
        aug(x, seed_base=0)
    torch.cuda.synchronize()
    gc.collect(); torch.cuda.empty_cache()

    print("\n=== Per-sample CPU (Dataset.__getitem__ style) ===")
    t_cpu = bench_per_sample_cpu(aug)
    t_cpu.sort()
    med_cpu = t_cpu[len(t_cpu) // 2]
    print(f"  median: {1000 * med_cpu:.0f} ms / batch  ({B / med_cpu:.0f} samples/s)")

    print("\n=== Batch GPU (train-loop style) ===")
    t_gpu = bench_batch_gpu(aug)
    t_gpu.sort()
    med_gpu = t_gpu[len(t_gpu) // 2]
    print(f"  median: {1000 * med_gpu:.0f} ms / batch  ({B / med_gpu:.0f} samples/s)")

    speedup = med_cpu / med_gpu
    print(f"\n→ Batch GPU is {speedup:.1f}× faster than per-sample CPU on this GPU.")


if __name__ == "__main__":
    main()
