# paraug

![paraug banner — CPU and GPU augmentation pipelines converging to a single bit-exact output](docs/banner.png)

> **Bit-exact CPU/GPU parity for image augmentation.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9-3.12](https://img.shields.io/badge/python-3.9--3.12-blue.svg)]()

**Languages**: English | [繁體中文](README_zh-TW.md)

`paraug` is a PyTorch-native augmentation library that guarantees **the same
seed produces the same output on CPU and CUDA**. Per-primitive RNG is sampled
on CPU regardless of tensor device, so a training run that randomly switches
between CPU and GPU stages — or a unit test that swaps backends — stays
deterministic.

## Why parity matters

Most augmentation libraries (albumentations, kornia, torchvision) use device-
local RNG. Same seed, different output across CPU/CUDA. This bites in three
places:

1. **Reproducibility**: paper-to-code lineage breaks when a reviewer can't
   match published numbers.
2. **Debugging**: CPU-side unit tests don't catch GPU-only bugs and vice
   versa.
3. **Distributed training**: workers on heterogeneous hardware drift apart.

paraug fixes this by isolating RNG to CPU (`torch.Generator(device="cpu")`)
and routing only the deterministic torch ops through device. Tolerance:

- **Elementwise ops** (gamma, noise, color jitter, …): atol 1e-6
- **`grid_sample`-class ops** (affine, perspective, tps, …): atol 2e-4
  (bilinear ulp drift across ATen vs cuDNN)

## Installation

```bash
pip install paraug
```

Or from source:

```bash
pip install git+https://github.com/alieuidsh/paraug.git
```

## Quickstart

```python
import torch
from paraug import AugPipeline

aug = AugPipeline({
    "geometric": {
        "affine": {"p": 1.0, "rot_deg": 15.0, "scale_range": (0.9, 1.1)},
        "tps":    {"p": 0.5, "max_disp": 12.0, "n_ctrl": 5},
    },
    "photometric": {
        "gamma":         {"p": 0.5},
        "color_jitter":  {"p": 0.5},
        "gaussian_blur": {"p": 0.3},
    },
})

img  = torch.rand(2, 3, 256, 256)         # (B, C, H, W)
mask = torch.ones(2, 1, 256, 256)         # optional

img_out, mask_out = aug(img, mask=mask, seed_base=42, epoch=0, step=0)
```

The same call on GPU is bit-exact within tolerance:

```python
img_gpu  = img.cuda()
mask_gpu = mask.cuda()
img_cuda, mask_cuda = aug(img_gpu, mask=mask_gpu, seed_base=42, epoch=0, step=0)
assert (img_out - img_cuda.cpu()).abs().max() < 2e-4
```

## Stacking extra spatial channels (GT-as-channel)

`n_image_channels=N` declares that the first N input channels are the
"image" (geometric + photometric) and any remaining channels follow
geometric warp only. Photometric primitives skip the extra channels, so
stacked ground-truth fields stay numerically intact while sharing the
exact back-warp grid as the image:

```python
import torch
from paraug import AugPipeline

# (B, 3, H, W) RGB + (B, 2, H, W) full-image heatmap GT = 5 channels.
img_rgb  = torch.rand(2, 3, 256, 256)
gt_h     = render_h_line_heatmap(...)   # your renderer; (B, 1, H, W)
gt_v     = render_v_line_heatmap(...)   # (B, 1, H, W)
img_5ch  = torch.cat([img_rgb, gt_h, gt_v], dim=1)   # (B, 5, H, W)

aug = AugPipeline({
    "geometric":   {"affine": {"p": 1.0, "rot_deg": 10.0},
                      "tps":    {"p": 0.5, "max_disp": 8.0, "n_ctrl": 5}},
    "photometric": {"gamma": {"p": 0.5, "gamma_range": (0.8, 1.2)}},
}, n_image_channels=3)

out, _ = aug(img_5ch, seed_base=42)
# out[:, :3] = warped + gamma-corrected RGB
# out[:, 3:] = warped (only) heatmap — gamma did NOT touch it
```

This eliminates a common pain point in tasks where GT is a 2-D field (line
heatmaps, segmentation masks with continuous labels, distance transforms,
tangent fields): instead of solving a separate forward-warp problem for
GT, render GT as image channels, stack, and let `grid_sample` warp
everything in one pass. The default `n_image_channels=None` preserves the
prior behaviour for callers that don't need the split.

`random_shadow` is geometric in dispatch but multiplicative in effect; the
split correctly treats it as photometric so extra channels are not dimmed
by the shadow factor.

### Sampling-mode note (`mask` vs extra channels)

Extra channels stacked onto `img` are sampled with **bilinear**
interpolation — same as the image. If you need **nearest** interpolation
(e.g. integer class labels or segmentation IDs that must not be
interpolated), pass that tensor as the `mask=` argument instead of
stacking it onto `img`:

| Path | Interp | Photometric applied? | Channel count |
|------|--------|----------------------|---------------|
| `img[:, :n_image_channels]` (RGB / image) | bilinear | yes | any |
| `img[:, n_image_channels:]` (extra) | bilinear | **no** | any |
| `mask` argument | **nearest** | no | 1 (single-channel) |

paraug warps `img` and `mask` with the same back-warp grid in every
geometric primitive — only the interpolation mode differs. Photometric
primitives never modify `mask`.

## Primitives

### Geometric (7)

| Name | Description |
|---|---|
| `affine` | Rotation + scale + translation via `F.affine_grid` |
| `perspective` | 4-point homography from corner jitter |
| `random_crop_pad` | Scale-then-pad crop, area-preserving |
| `elastic_transform` | Bilinear-upsampled random displacement field |
| `optical_distortion` | Radial barrel / pincushion (k·r²) |
| `random_shadow` | Soft-blurred triangle multiplicative shadow |
| `tps` | Thin-plate-spline-like warp from low-res control grid |

### Photometric (24)

Intensity / color: `gamma`, `color_jitter`, `hue_shift`, `random_grayscale`,
`lighting`, `clahe`, `local_contrast`, `sharpness`.

Noise: `gaussian_noise`, `salt_pepper_noise`, `salt_patches`.

Blur / artifacts: `gaussian_blur`, `motion_blur`, `jpeg_approx`.

Lighting: `vignette`, `specular_highlight`, `specular_streaks`.

Content overlays: `cutout`, `paper_texture_overlay`, `watermark`,
`random_text_overlay`, `background_compose`, `stains`, `creases`.

## Parity comparison

| Library | Bit-exact CPU↔GPU | Per-item RNG | GPU native | Mask-aware | Batch-native | # Geometric¹ | # Photometric¹ | License |
|---|---|---|---|---|---|---|---|---|
| **paraug** | **✓** (1e-6 / 2e-4)² | ✓ | ✓ (torch) | ✓ | ✓ | 7 | 24 | Apache 2.0 |
| albumentations | ✗ (numpy-only) | ✓ | ✗ | ✓ | partial | ~20 | ~50+ | MIT |
| kornia | ✗ (device-local RNG) | ✓ | ✓ (torch) | ✓ | ✓ | ~10 | ~45 | Apache 2.0 |
| torchvision.v2 | ✗ (device-local RNG) | ✓ | ✓ (torch) | partial | ✓ | ~18 | ~12 | BSD-3 |
| imgaug | ✗ (numpy-only) | ✓ | ✗ | ✓ | partial | ~20 | ~40 | MIT |
| augly | ✗ (PIL-only) | ✓ | ✗ | ✗ | ✗ | ~5 | ~20 | MIT |

<sub>¹ External counts are approximate as of 2026-05 (sampled from each project's
`__init__.py` / docs index). Versions move fast — consult each project's
authoritative API reference for current numbers. paraug counts are
code-exact (`len(GEOMETRIC_PRIMITIVES)` / `len(PHOTOMETRIC_PRIMITIVES)`).</sub>

<sub>² Tolerance verified by `tests/test_parity.py` on NVIDIA 5060 Ti + 4080
at v0.1.0; exact bounds: 1e-6 for the 6 elementwise photometric ops listed
in `PHOTO_ELEMENTWISE` (gamma / gaussian_noise / color_jitter / vignette /
cutout / hue_shift), 2e-4 for grid_sample-class (geometric) and conv-class
(blur) ops. **GitHub free CI runners are CPU-only**, so the 13 CUDA parity
tests skip on CI — community verification on additional GPU SKUs is
welcome (open a PR with the result, or run `pytest tests/test_parity.py -k
cpu_vs_cuda` locally and post the output).</sub>

### When to use paraug

- Cross-device reproducibility (paper-grade ablation where CPU↔GPU drift breaks a baseline)
- Distributed training on heterogeneous hardware
- Unit-test-friendly augmentation pipelines (CPU-side RNG means a test on a free CI runner reproduces a developer's GPU result)

### When **NOT** to use paraug

- You need 50+ primitive options out of the box → try `albumentations` or `imgaug`
- You need PIL-style per-image API → try `augly`
- You need built-in compositional ops like `OneOf` / `SomeOf` → try `albumentations`

## Examples

See `examples/`:

- `01_quickstart.py` — minimal load → augment → save
- `02_mask_aware.py` — image + segmentation mask warped together
- `03_cpu_gpu_parity.py` — same seed on CPU and CUDA, assert
  `max_abs_diff < 2e-4`

## Citation

```bibtex
@software{paraug2026,
  author = {alieuidsh},
  title  = {paraug: Bit-exact CPU/GPU parity for image augmentation},
  year   = {2026},
  url    = {https://github.com/alieuidsh/paraug},
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
