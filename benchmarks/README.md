# paraug benchmarks

Self-benchmark of paraug throughput across CPU and CUDA on a representative
augmentation pipeline. **No cross-library comparison** — that requires
matching primitive semantics across implementations and is deferred to v0.2.

## Pipeline

The benchmark exercises a mixed geometric + photometric pipeline that hits
all three tolerance classes (matrix-based, grid-sample, conv):

```python
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
```

Five primitives in fixed order. All `p=1.0` so every iteration hits every op.

## Methodology

- **Warmup**: 5 iterations excluded from timing — amortises first
  `grid_sample` JIT specialisation, CUDA kernel cache, and TPS control-grid
  upsample setup.
- **Measurement**: 20 iterations, **median ms/batch** reported. p5/p95 also
  shown to indicate jitter.
- **Throughput**: `(batch_size × 1000) / median_ms` images/sec.
- **CUDA sync**: `torch.cuda.synchronize()` around every iteration so we
  measure end-to-end batch latency, not async kernel-launch return time.
- **RNG**: per-item seed advances each iteration so the GPU doesn't reuse
  cached random buffers (more representative of real training).

## Results

### Reference: NVIDIA RTX 5060 Ti 16GB, torch 2.12 cu130, Windows 11

CPU is single-process (no `torch.set_num_threads` tuning); GPU uses one device.

| Device | Image | Batch | Median ms | Throughput (imgs/s) |
|---|---:|---:|---:|---:|
| CPU | 256×256 | 1 | 7.3 | 138 |
| CPU | 256×256 | 8 | 80.2 | 100 |
| CPU | 256×256 | 32 | 322.9 | 99 |
| CPU | 512×512 | 1 | 26.9 | 37 |
| CPU | 512×512 | 8 | 256.9 | 31 |
| CPU | 512×512 | 32 | 949.2 | 34 |
| CPU | 1024×1024 | 1 | 107.8 | 9 |
| CPU | 1024×1024 | 8 | 934.3 | 9 |
| CPU | 1024×1024 | 32 | 3995.8 | 8 |
| **CUDA** | 256×256 | 1 | 3.3 | 305 |
| **CUDA** | 256×256 | 8 | 17.3 | 462 |
| **CUDA** | 256×256 | 32 | 60.2 | **532** |
| **CUDA** | 512×512 | 1 | 9.4 | 106 |
| **CUDA** | 512×512 | 8 | 45.7 | 175 |
| **CUDA** | 512×512 | 32 | 213.8 | 150 |
| **CUDA** | 1024×1024 | 1 | 29.0 | 35 |
| **CUDA** | 1024×1024 | 8 | 202.1 | 40 |
| **CUDA** | 1024×1024 | 32 | 775.2 | 41 |

CPU→CUDA speedup at large batch:

| Image | CPU bs=32 | CUDA bs=32 | Speedup |
|---|---:|---:|---:|
| 256×256 | 99 imgs/s | 532 imgs/s | **5.4×** |
| 512×512 | 34 imgs/s | 150 imgs/s | **4.4×** |
| 1024×1024 | 8 imgs/s | 41 imgs/s | **5.1×** |

Raw JSON: [`results_5060ti.json`](results_5060ti.json).

## How to reproduce

```bash
pip install paraug
git clone https://github.com/alieuidsh/paraug.git
cd paraug
python benchmarks/paraug_self.py
```

The script writes results to stdout and (with `--json out.json`) to disk.
See `python benchmarks/paraug_self.py --help` for options:

```
--sizes 256 512 1024      # image sizes (square)
--batches 1 8 32          # batch sizes
--device cpu|cuda|both    # which to benchmark
--warmup 5                # warmup iterations
--iter 20                 # measurement iterations
--seed 42                 # base seed
--json out.json           # also write JSON
```

## Adding your hardware

Community contributions of additional hardware results are welcome. Open a
PR adding a `results_<hwname>.json` from `paraug_self.py --json` and append
a row to this README's reference table noting:

- Torch version
- CUDA/ROCm version (or "CPU only")
- Driver version (optional)
- OS

Please **don't** modify the pipeline config — keep `DEFAULT_CFG` so cross-
hardware comparison stays apples-to-apples.

## What this benchmark does NOT measure

- **Cross-library comparison**. albumentations / kornia / torchvision /
  imgaug / augly have different primitive semantics and runtime models;
  fair comparison needs matched configs and per-library warmup
  protocols. Planned for v0.2.
- **Memory footprint**. `torch.cuda.memory_allocated()` snapshots could be
  added in a future iteration.
- **Per-primitive cost**. Right now we time the whole pipeline; per-op
  attribution would require more invasive instrumentation.
- **Multi-worker DataLoader scaling**. paraug runs inside the worker, so
  `num_workers > 1` typically dominates throughput on a real training run;
  this benchmark only measures single-call cost.
