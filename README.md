# paraug

![paraug banner — CPU and GPU augmentation pipelines converging to a single bit-exact output](docs/banner.png)

> **Bit-exact CPU/GPU parity for image augmentation.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9-3.12](https://img.shields.io/badge/python-3.9--3.12-blue.svg)]()

**Languages**: English | [繁體中文](README_zh-TW.md)

`paraug` is a general-purpose PyTorch-native augmentation library.
**35 primitives** (7 geometric + 28 photometric, including CutMix /
MixUp / GridMask / RandomErasing), **GPU-batch-native**, with
**bit-exact CPU/GPU parity**: the same seed produces the same output
on CPU and CUDA. Per-primitive RNG is sampled on CPU regardless of tensor
device, so a training run that randomly switches between CPU and GPU
stages — or a unit test that swaps backends — stays deterministic.

It's a drop-in replacement for `kornia.augmentation` / `torchvision.v2`
when you need reproducibility across heterogeneous hardware, and a
batch-native alternative to `albumentations` when you want GPU acceleration.

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

# Build your own config from the 31 primitives. Per-op `p` is independent
# (each op fires with its own probability per sample).
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

# Input expectations: float tensor in [0, 1], shape (B, C, H, W).
# (Numpy HWC uint8 is also accepted; paraug normalises internally.)
img  = torch.rand(2, 3, 256, 256)         # (B, C, H, W) in [0, 1]
mask = torch.ones(2, 1, 256, 256)         # optional segmentation mask

# aug always returns a (img, mask) tuple — discard with `_` if no mask:
img_out, mask_out = aug(img, mask=mask, seed_base=42, epoch=0, step=0)
img_only, _      = aug(img,             seed_base=42, epoch=0, step=0)
```

Same call on GPU is bit-exact within tolerance:

```python
img_cuda, mask_cuda = aug(img.cuda(), mask=mask.cuda(),
                           seed_base=42, epoch=0, step=0)
assert (img_out - img_cuda.cpu()).abs().max() < 2e-4
```

### `seed_base`, `epoch`, `step`

These three integers compose into the per-item RNG seed (along with the
item's batch position). Same triple → same output for that item.

- **`seed_base`** — run-level seed. Pin this in your config; reuse across
  the whole training run.
- **`epoch`** — change across epochs so the same dataset sample gets
  different augmentation each pass.
- **`step`** — change within an epoch so successive batches of the same
  underlying dataset position (rare; usually `step = global_step`) don't
  collide.

For inference / one-shot use, all three may be 0 (`aug(img, seed_base=0)`).
The split exists so training-time augmentation is reproducible **and**
varies along the right axes; you don't have to use all three.

## Where to put paraug in your training code

The first instinct, transferred from `torchvision.transforms`, is to put
augmentation inside `Dataset.__getitem__` so each worker processes one
sample at a time. **Don't do this with paraug** — it's a batch-native GPU
library, and per-sample CPU placement throws away the GPU acceleration.

```python
# ❌ DON'T — per-sample CPU augmentation in worker processes
class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.aug = AugPipeline(cfg)
    def __getitem__(self, idx):
        img = load_image(idx)                    # (C, H, W), CPU
        img, _ = self.aug(img.unsqueeze(0), seed_base=idx)   # CPU aug
        return img.squeeze(0)

# ✅ DO — Dataset loads only, train loop augments the GPU batch
class MyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return load_image(idx)                   # just I/O + resize

aug = AugPipeline(cfg, canvas_size=(224, 224))
for step, (images, labels) in enumerate(loader):
    images = images.to(device, non_blocking=True)
    images, _ = aug(images, seed_base=42, epoch=epoch, step=step)
    logits = model(images)
    ...
```

Measured on a 5060 Ti at bs=32 canvas=224×224 (rough config of 5 ops):

| Placement | Wall time / batch | Throughput |
|---|---:|---:|
| Per-sample CPU in Dataset | 219 ms | 146 samples/s |
| **Batch GPU in train loop** | **75 ms** | **429 samples/s** |

→ **2.9× speedup** just from moving augmentation to the right place. The
gap grows with batch size, canvas size, and op count (paraug's per-op
launch overhead is amortised across the batch).

For larger-batch / larger-canvas setups, also see
[`set_fast_noise(True)`](#performance-tuning-fast-noise-and-chunk_size)
and [`chunk_size`](#performance-tuning-fast-noise-and-chunk_size) below.

## Compositing: `compose(foreground, background, mask)`

`compose` blends a foreground onto a background through a mask, then runs
the configured aug:

```python
from paraug import AugPipeline

aug = AugPipeline({
    "geometric":   {"affine": {"p": 1.0, "rot_deg": 10.0}},
    "photometric": {"gamma": {"p": 0.5}},
})

# numpy (H, W, 3) uint8 in → numpy out  (also accepts torch tensors)
img, mask = aug.compose(
    foreground = paper_image,   # the sheet to paste
    background = scene_image,   # the static backdrop
    mask       = paper_mask,    # 255 = foreground, 0 = background
)
```

Data flow:

1. **geometric** primitives warp `(foreground, mask)` together — the
   foreground sheet rotates / scales / warps while the background stays
   put.
2. **blend** — `composite = fg_w * mask_w + background * (1 - mask_w)`.
3. **photometric** primitives perturb the composite.
4. optional **`canvas_size`** stretch (see below).

**Layered synthesis** is just two `compose` calls — pass-1 output becomes
pass-2's foreground:

```python
# "content printed on paper, then paper photographed in a scene"
img1, m1 = aug.compose(content, paper_tone, content_mask)   # printing
img2, m2 = aug.compose(img1,    scene_bg,   paper_mask)      # photographing
```

Use two `AugPipeline` instances if the two passes need different aug.

## Fixed output size: `canvas_size`

```python
aug = AugPipeline(config, canvas_size=(512, 512))
```

Every `__call__` / `compose` output is stretched to `(512, 512)` with a
non-uniform `F.interpolate` — input aspect ratio is **not** preserved.
This is the right choice when downstream batching needs uniform shapes
and the task is consistent under stretch (train and inference both
stretch to the same canvas, so the model learns in canvas space).
Default `None` keeps the output size equal to the input.

Ground truth carried **inside** the tensor — the `mask`, or channels
stacked via `n_image_channels` — is stretched alongside the image for
free. For GT stored as coordinates **outside** the tensor, pass
`return_transform=True` to `compose` and rescale with the returned
`scale_x` / `scale_y`:

```python
img, mask, t = aug.compose(fg, bg, m, return_transform=True)
line_x = [x * t["scale_x"] for x in line_x]
line_y = [y * t["scale_y"] for y in line_y]
```

## Nested-frame layout: `place_into_canvas`

When the segmentation target is a sub-region of a larger frame — and the
outer frame is itself rectangular — the model can latch onto the outer
rectangle as a shortcut. Random layout at training time forces it to
learn that the wider surrounding frame is a *distractor*.

`place_into_canvas` embeds a foreground (and its mask) at a random
position inside a larger constant-colour canvas, with random per-axis
margins:

```python
from paraug import place_into_canvas

# content: (H, W, 3) uint8 — the inner region you actually want to segment
# content_mask: (H, W) uint8 — segmentation target
padded, padded_mask = place_into_canvas(
    content, content_mask,
    canvas_size=(800, 1000),
    fill=(245, 245, 245),               # background colour
    margin_frac_range=(0.05, 0.30),     # 5-30% margin per side, randomised
    seed_base=epoch_step_seed,
)
```

The deterministic CPU-side per-item RNG (same `seed_base / epoch / step`
convention as the primitives) makes every batch position bit-exactly
reproducible across CPU and CUDA.

## Performance tuning: `fast_noise` and `chunk_size`

Two opt-in knobs trade a small contract for a large speed / VRAM win on
GPU; both default to off so behaviour matches the docs above for callers
that don't set them.

### `paraug.set_fast_noise(True)` — speed

Switches the three CPU-sample noise primitives (`gaussian_noise`,
`jpeg_approx`, `salt_pepper_noise`) to GPU-side `torch.randn` /
`torch.rand`. The CPU-path noise tensor is the per-call wall-time hot
spot at large canvases (~350 ms at bs=20 canvas=1024) because it's
filled in a Python per-item loop and copied to GPU; the GPU path takes
~6 ms. **Measured ~1.85× end-to-end speedup** on a 5060 Ti at bs=20
canvas=1024 with a 14-op pipeline.

Contract: cuRAND ≠ MT19937, so `fast_noise=True` produces different
output than `fast_noise=False` for the same seed. Determinism per
`(seed_base, epoch, step)` is preserved within either mode. Leave off
when running the parity tests; turn on for production training.

### `paraug.set_device_rng(True)` — speed *without* giving up parity

Same three noise primitives, but the dense field is generated on the
device by a counter-based **Philox4x32-10** written in integer torch ops
(`paraug/philox.py`), so the uint32 stream is bit-identical on CPU and
CUDA (Random123 known-answer vectors pass) and item `i` never depends on
which other items share the batch. Env: `PARAUG_PHILOX=1`,
`PARAUG_DEVICE_RNG=philox|hash`.

```python
paraug.set_device_rng(True)                  # Philox4x32-10
paraug.set_device_rng(True, backend="hash")  # lowbias32: ~2.5x less memory traffic
```

Contract: sample values differ from the CPU-generator stream (same
distribution), so it is opt-in and runs made without it reproduce exactly.
Precedence when both flags are on: `fast_noise` > `device_rng` > CPU.
Only enable on CUDA — the same integer path on CPU is slower than
MT19937. Measured with every primitive at p=1, bs32 256²: RTX 4090
97 → 222 img/s, RTX 3060 83 → 128 img/s (hash).

### `AugPipeline(cfg, ..., chunk_size=N)` — VRAM

Splits the batch into sub-batches of size `N` internally, runs the full
pipeline on each, concatenates outputs. Per-call peak alloc scales with
`N` instead of batch size. **30-40% peak alloc reduction** at bs=20 →
chunk_size=5, with no wall-clock penalty (cache hits between primitives
offset the per-chunk launch overhead).

Contract: chunked output is deterministic per
`(seed_base, epoch, step, chunk_size)` but the per-item seed namespace
shifts when you change `chunk_size`, so don't expect bit-equality if you
toggle it mid-run.

## Optional presets

`paraug.presets` ships hand-tuned configs for common deployment scenarios.
Each preset returns a deep-copyable dict you can adjust:

```python
from paraug import AugPipeline, presets
cfg = presets.OOD_PRINTED_PAPER()         # one current preset; more may follow
aug = AugPipeline(cfg, canvas_size=(512, 512))
```

Presets are **not** the primary API — build your own config from the 31
primitives (Quickstart above) for any task that doesn't match a preset
exactly. See `paraug/presets.py` for what each preset contains and
`examples/05_ood_printed_paper.py` for a full layered-synthesis example.

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

### Photometric (28)

Intensity / color: `gamma`, `color_jitter`, `hue_shift`, `random_grayscale`,
`lighting`, `clahe`, `local_contrast`, `sharpness`.

Noise: `gaussian_noise`, `salt_pepper_noise`, `salt_patches`.

Blur / artifacts: `gaussian_blur`, `motion_blur`, `jpeg_approx`,
`defocus_blur`.

Lighting / glare: `vignette`, `specular_highlight`, `specular_streaks`,
`paper_glare`.

Colour cast / WB: `spatial_color_cast`, `white_balance_shift`.

Content overlays: `cutout`, `paper_texture_overlay`, `watermark`,
`random_text_overlay`, `background_compose`, `stains`, `creases`.

Region drop / pair mixing (v0.7.0): `random_erasing`, `grid_mask`,
`cutmix`, `mixup`. See [CutMix / MixUp label mixing](#cutmix--mixup-label-mixing)
below for the `paraug.mix_info` helper that recovers the per-item
(λ, partner_idx) for classification training.

### CutMix / MixUp label mixing

`cutmix` and `mixup` mix pairs of items in a batch. For classification
training the *labels* need to be mixed by the same λ paraug used
internally. paraug doesn't track labels — `paraug.mix_info()` recovers
the (λ, partner_idx, gate) tensors using the same seed:

```python
import paraug

aug = paraug.AugPipeline({"photometric": {
    "cutmix": {"p": 1.0, "alpha": 1.0},
}})

for step, (images, labels) in enumerate(loader):
    images = images.to(device)
    images, _ = aug(images, seed_base=step, epoch=epoch, step=step)
    # Recover λ + partner using the same triple paraug used:
    info = paraug.mix_info("cutmix", seed_base=step, epoch=epoch,
                            step=step, B=images.shape[0],
                            p=1.0, alpha=1.0)
    labels_a = labels                       # original labels
    labels_b = labels[info["perm"]]         # partner labels
    lam = info["lam"].to(device)            # (B,) sampled λ
    # mix labels per-item; only items where info["gate"] is True
    # actually got cutmix-augmented.
    # ...your loss code...
```

For exact area-ratio λ after rounding (cutmix only), pass
`primitive="cutmix_actual_lam"` and `img_shape=(B, C, H, W)`.

### Inspecting spec keys: `paraug.describe(name)`

Every primitive accepts a `{"p": ..., ...primitive-specific keys...}`
spec dict. To find the spec keys (and their defaults) for any primitive
without grep-ing the source, call `paraug.describe`:

```python
>>> import paraug
>>> paraug.describe("affine")
affine (geometric)
==================
Random rotation / scale / translation.

    spec = {"p": prob, "rot_deg": float, "scale_range": (lo, hi),
            "translate_frac": float (fraction of H/W)}

spec keys (with defaults):
  scale_range            = (0.85, 1.15)
  p                      = 1.0
  rot_deg                = 30.0
  translate_frac         = 0.05

>>> paraug.describe()            # one-line summary of every primitive
>>> info = paraug.describe("affine", return_dict=True)   # programmatic
>>> info["spec_keys"]
{'scale_range': (0.85, 1.15), 'p': 1.0, 'rot_deg': 30.0, 'translate_frac': 0.05}
```

Defaults are extracted by AST walk of each primitive function's
`spec.get(...)` calls, so they stay in sync with the implementation.

## Parity comparison

| Library | Bit-exact CPU↔GPU | Per-item RNG | GPU native | Mask-aware | Batch-native | # Geometric¹ | # Photometric¹ | License |
|---|---|---|---|---|---|---|---|---|
| **paraug** | **✓** (1e-6 / 2e-4)² | ✓ | ✓ (torch) | ✓ | ✓ | 7 | 28 | Apache 2.0 |
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

## Migration from torchvision / albumentations / kornia

paraug uses a dict-based config instead of the `Compose([...])` flat list,
but the underlying ops are the same. Direct equivalents:

| torchvision.transforms.v2 | albumentations | kornia.augmentation | **paraug** |
|---|---|---|---|
| `RandomAffine` | `Affine` | `RandomAffine` | `affine` |
| `RandomPerspective` | `Perspective` | `RandomPerspective` | `perspective` |
| `ElasticTransform` | `ElasticTransform` | `RandomElasticTransform` | `elastic_transform` |
| `RandomResizedCrop` | `RandomResizedCrop` | `RandomResizedCrop` | `random_crop_pad` + `canvas_size` |
| `ColorJitter` | `ColorJitter` | `ColorJitter` | `color_jitter` |
| `RandomGrayscale` | `ToGray` | `RandomGrayscale` | `random_grayscale` |
| `GaussianBlur` | `GaussianBlur` | `RandomGaussianBlur` | `gaussian_blur` |
| `GaussianNoise` | `GaussNoise` | `RandomGaussianNoise` | `gaussian_noise` |
| `RandomErasing` | `CoarseDropout` | `RandomErasing` | `random_erasing` |
| (none) | `GridDropout` | `RandomGridShuffle` (no equiv) | `grid_mask` |
| `CutMix` | (none) | `RandomCutMixV2` | `cutmix` (+ `paraug.mix_info`) |
| `MixUp` | (none) | `RandomMixUpV2` | `mixup` (+ `paraug.mix_info`) |
| `HorizontalFlip` / `VerticalFlip` | `HorizontalFlip` | `RandomHorizontalFlip` | use `affine` with `rot_deg=0` (paraug doesn't have a separate flip yet — open an issue if you need one) |

Example: a typical classification pipeline rewritten

```python
# torchvision.transforms.v2 style:
#     transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(0.4, 0.4, 0.4),
#         transforms.RandomErasing(p=0.25),
#         transforms.ToTensor(),
#     ])

# paraug equivalent (canvas_size handles resize; flip TBD — affine is the closest match):
import paraug
aug = paraug.AugPipeline({
    "geometric":   {"affine":         {"p": 1.0, "rot_deg": 5.0, "scale_range": (0.8, 1.0)}},
    "photometric": {"color_jitter":   {"p": 1.0, "brightness": 0.4, "contrast": 0.4, "saturation": 0.4},
                    "random_erasing": {"p": 0.25, "size_frac_range": (0.02, 0.2)}},
}, canvas_size=(224, 224))

# Train loop — apply on the GPU batch, NOT inside Dataset.__getitem__:
for step, (images, labels) in enumerate(loader):
    images = images.to(device, non_blocking=True)
    images, _ = aug(images, seed_base=42, epoch=epoch, step=step)
    logits = model(images)
    # ...
```

See [`Where to put paraug`](#where-to-put-paraug-in-your-training-code)
for why the train-loop placement (vs `Dataset.__getitem__`) matters —
paraug is GPU-batch-native, and per-sample CPU placement throws away
2-3× speedup.

## Examples

See `examples/`:

- `01_quickstart.py` — minimal load → augment → save
- `02_classification.py` — Dataset + DataLoader + train loop with
  batch-GPU augmentation (the "Where to put paraug" pattern, end-to-end)
- `02_mask_aware.py` — image + segmentation mask warped together
- `03_cpu_gpu_parity.py` — same seed on CPU and CUDA, assert
  `max_abs_diff < 2e-4`
- `04_compose_layered.py` — two-pass `compose` for layered synthesis
- `05_ood_printed_paper.py` — `OOD_PRINTED_PAPER` preset, full pipeline

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
