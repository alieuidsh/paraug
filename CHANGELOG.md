# Changelog

All notable changes to paraug are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2026-09-05

### Added — device-side RNG that keeps parity

- **`paraug.set_device_rng(True, backend="philox")`** (env `PARAUG_PHILOX=1`,
  `PARAUG_DEVICE_RNG=philox|hash`) — third option between the CPU generator
  (bit-exact, slow) and `set_fast_noise` (cuRAND, fast, breaks parity).
  `paraug/philox.py` implements **Philox4x32-10** in pure integer torch ops
  (16-bit split multiply, no int64 overflow), so the uint32 stream is
  bit-identical on CPU and CUDA (Random123 known-answer vectors pass) while
  `gaussian_noise` / `jpeg_approx` / `salt_pepper_noise` generate their dense
  fields on the device. Per-item keyed → item `i` is independent of batch
  composition (unlike `fast_noise`, which seeds one generator per batch).
  Per-item gate/sigma still come from the CPU generator. Precedence:
  `fast_noise` > `device_rng` > CPU. `backend="hash"` (lowbias32) uses ~2.5x
  less memory traffic and wins on bandwidth-limited GPUs (RTX 3060).
- `tests/test_philox.py` — KAT vectors, mulhilo vs Python bigint, uniform /
  normal stats, batch-size independence, CPU↔CUDA bit-exactness, noise-op
  parity in both backends, and a guard that the default mode is unchanged.

### Changed — batched dense compute on device

- `perspective`, `elastic_transform`, `creases`, `paper_texture_overlay`:
  scalar params are still sampled per item on CPU (semantics unchanged); the
  H×W work (homography warp, bilinear upsample, line rendering, z-normalise)
  is now one batched op on the image device instead of a per-item CPU loop
  plus full-resolution H2D copies. Outputs differ from 0.7.0 only by float
  ordering (≤ 5e-5 grid class, ≤ 1.2e-7 elementwise). perspective 55 → 11 ms,
  creases 46 → 2 ms, elastic 40 → 1.4 ms, paper_texture 22 → 0.8 ms
  (bs32 256², RTX 4090).

### Measured

All primitives at p=1, bs32 256², img/s: **RTX 4090 97 → 222** (philox),
**RTX 3060 83 → 128** (hash). End-to-end with a 6M-param net at 87% VRAM:
aug-inline step 824 → 644 ms. Caveat: the device-RNG path on *CPU* tensors
is 3-5x slower than MT19937 — only enable on CUDA.

## [0.7.0] - 2026-05-27

### Added — 4 modern aug primitives

- **`random_erasing`** — Zhong et al. 2017. Rectangular region replaced
  with sampled noise (`normal` / `uniform` / `constant` fill modes) at a
  per-item randomised position and aspect ratio. More aggressive than
  the existing `cutout` (constant fill).
- **`grid_mask`** — Chen et al. 2020. Drop a regular grid of small square
  regions across the whole image. Spec includes `ratio` (drop fraction
  per cell), `d_range` (cell size in px), and `rotation_deg` (max grid
  rotation per item).
- **`cutmix`** — Yun et al. 2019. Cut a rectangular region from a
  batch-partner image B and paste it into image A. Mixing fraction
  λ ~ Beta(α, α). Partner is `perm[i]` for sample i, where `perm` is a
  per-call randperm of the batch (paraug picks it deterministically from
  `seed_base / epoch / step`).
- **`mixup`** — Zhang et al. 2017. Linear interpolation
  `out[i] = λ·img[i] + (1-λ)·img[perm[i]]`, λ ~ Beta(α, α), clipped to
  [0.01, 0.99].

### Added — `paraug.mix_info(...)` helper

CutMix and MixUp need labels mixed by the same λ paraug used. paraug
doesn't track labels — `mix_info("cutmix" | "mixup", seed_base, epoch,
step, B, p=..., alpha=...)` returns `{"lam": (B,) float, "perm": (B,)
long, "gate": (B,) bool}` so the train loop can mix labels itself.
`mix_info("cutmix_actual_lam", ..., img_shape=(B, C, H, W))` returns the
post-rounding area ratio for tight CutMix reproduction.

### Added — migration guide

README has a "Migration from torchvision / albumentations / kornia"
section mapping common ops to paraug equivalents, with a worked
classification-pipeline rewrite.

### Compatibility

- No change to existing primitives or AugPipeline API.
- 4 new primitives auto-included in `paraug.photometric.list_primitives()`
  (28 photometric total, was 24).
- 24 new tests in `test_v070_modern_aug.py` including CPU/GPU parity for
  all 4 primitives (146 → 170 tests).

## [0.6.2] - 2026-05-25

### Added

- **`paraug.describe(name)` / `paraug.describe()`** — primitive introspection
  helper. Prints (or returns as a dict) the docstring plus the spec keys
  with their default values, extracted by AST walk of each primitive
  function's `spec.get(...)` calls (always in sync with the implementation).
  No more grep-ing the source to find out what spec keys a primitive
  accepts; `paraug.describe("affine")` prints the answer.
- **`examples/02_classification.py`** — end-to-end Dataset + DataLoader +
  train-loop with batch-GPU augmentation. The "Where to put paraug" pattern
  from the README, runnable with synthetic data so no dataset download.

### Changed (documentation only — no behaviour change)

- **README rewritten** to lead with the general-purpose framing (31
  primitives, GPU-batch-native, bit-exact CPU/GPU parity) instead of any
  specific use case. New sections:
  - "Where to put paraug in your training code" — explicit DO / DON'T
    with a 2.9× speedup benchmark for batch-GPU vs per-sample-CPU
    placement on a 5060 Ti at bs=32 canvas=224×224.
  - "Performance tuning: `fast_noise` and `chunk_size`" — both opt-in
    flags now have a dedicated subsection with the contract and the
    measured win.
  - Quickstart updated to show the `img, _ = aug(...)` tuple-discard
    pattern explicitly, document the input dtype/range expectation
    (`float in [0, 1], (B, C, H, W)`), and explain the
    `seed_base / epoch / step` triple.
- Preset section demoted to a small "Optional presets" pointer — presets
  are no longer documented as the primary API.
- Several docstrings and comments cleaned of domain-specific examples
  (e.g. `layout.py`, `pipeline.compose` docstring, `presets.py` comments)
  to keep the public surface neutral.
- Chinese README (`README_zh-TW.md`) mirrors the English rewrite.

## [0.6.0] - 2026-05-25

### Added

- **`paraug.geometric.list_primitives()`** / **`paraug.photometric.list_primitives()`** —
  return sorted list of primitive names available in this build. Useful for
  callers that want to enumerate the op space (e.g. building a config dict
  programmatically, or running an op-coverage sweep). Sorted ordering keeps
  cross-machine indexing deterministic.
- `paraug.geometric` and `paraug.photometric` submodules are now re-exported
  at the package level: `paraug.geometric.list_primitives()`.

### Changed

- `AugPipeline(config)` now accepts `config=None` and defaults to an empty
  dict, which gives an effective no-op pipeline. Convenient for callers
  that subclass `AugPipeline` and register primitives outside the dict
  path, or that want a placeholder pipeline before populating it.

### Yanked

- **`0.5.5` has been yanked from PyPI** — that revision added a feature
  that, on review, fell outside paraug's intended scope as a generic
  image-augmentation library and was better factored as caller-side code
  rather than library API. `0.6.0` keeps the small packaging niceties
  from that revision (listed above) and drops the rest. Existing
  `pip install paraug==0.5.5` installs keep working; new installs should
  use `0.6.0`.

## [0.5.4] - 2026-05-25

### Added

- **`AugPipeline(cfg, ..., chunk_size=N)`** — caps per-call VRAM peak by
  slicing the batch into sub-batches of size N internally and running the
  full pipeline on each, concatenating outputs. **Measured on home 5060 Ti
  at canvas 1024 bs=20 with `OOD_PRINTED_PAPER` preset + `fast_noise=True`**:

  | chunk_size | peak_alloc | peak_reserved | wall (median ms) |
  |------------|------------|---------------|------------------|
  | None (B=20)| 2.42 GB    | 3.17 GB       | 503              |
  | 10         | 1.68 GB    | 2.36 GB       | 484              |
  | 5          | 1.50 GB    | 2.04 GB       | 480              |
  | 4          | 1.47 GB    | 1.81 GB       | 475              |

  Peak alloc drops 30-40% with no wall-clock cost (cache hits between
  primitives offset the per-chunk launch overhead). Use this when a large
  effective batch fits the model forward but not the aug-side peak —
  e.g. a 60 M-param segmentation model on a 16 GB GPU at bs=20 OOMs on
  aug intermediates before model fwd even starts.

  **Output determinism contract**: chunk_size=N gives reproducible output
  for the same `(seed_base, epoch, step, chunk_size)`, but the per-item
  seed mapping shifts when you change chunk_size — items at local position
  0 within each chunk get a chunk-offset seed so cross-chunk items don't
  collide. Equivalent training-data distribution; just don't expect bit-
  exact output if you toggle chunk_size mid-run.

### Changed

- Photometric `_xy_coords` (pixel-coord meshgrid used by ~10 primitives)
  now cached at module level by `(H, W, dtype, device)`. Cheap — 16 MB per
  entry at canvas 1024 — but eliminates ~10 redundant 8 MB alloc/free per
  `compose` call. Call `paraug.photometric.clear_xy_cache()` if you change
  canvas size mid-run and want to drop the old entries.
- `defocus_blur` now `del padded` between the two separable convs so the
  reflect-padded copy (~257 MB at bs=20 canvas=1024 sigma=3.5) frees
  before the second conv's intermediate.

### Compatibility

- Default `chunk_size=None` preserves v0.5.3 behavior exactly.
- Three new tests in `test_compose.py`:
  `test_chunk_size_correctness_deterministic_per_seed`,
  `test_chunk_size_distinct_items_across_chunks`,
  `test_chunk_size_unchunked_matches_when_batch_fits`.

## [0.5.3] - 2026-05-25

### Added

- **`paraug.set_fast_noise(True)`** — opt-in module flag that switches the
  three hottest noise primitives (`gaussian_noise`, `jpeg_approx`,
  `salt_pepper_noise`) to GPU-side `torch.randn` / `torch.rand` instead of
  the default CPU-sample + CPU→GPU copy. **Measured 1.85× end-to-end speedup**
  on the `OOD_PRINTED_PAPER` preset at bs=20 canvas=1024 (936 → 506 ms
  median per `aug.compose` call on a 5060 Ti). Default OFF — existing
  CPU/GPU bit-exact parity tests in `test_parity.py` keep passing unchanged.
  Trade-off: cuRAND ≠ MT19937, so `fast_noise=True` produces different output
  than `fast_noise=False` for the same seed. Determinism per
  `(seed_base, epoch, step)` is preserved within either mode. Use in
  production training where seed-deterministic is enough; leave off when
  running parity tests or comparing against a CPU-aug baseline.

### Compatibility

- **No API change** for existing callers. Default state matches v0.5.2 exactly.
- Three new tests in `test_photometric.py`:
  `test_fast_noise_default_off_preserves_parity`,
  `test_fast_noise_deterministic_per_seed`,
  `test_fast_noise_breaks_cpu_path_parity_by_design`.

## [0.5.2] - 2026-05-24

### Fixed

- **GPU VRAM blowup when callers pass graph-tracked tensors** — `AugPipeline.__call__`
  and `AugPipeline.compose()` now wrap their bodies in `torch.no_grad()`.
  Augmentation is preprocessing and is never backpropped through, so this is
  defensive and semantically a no-op for correct callers. **Why it matters**:
  a downstream training loop that passes aug a tensor still attached to a
  live autograd graph (e.g. forgot to `.detach()` after a TPS warp built with
  `F.grid_sample`) was forcing every primitive's intermediate tensor to stay
  alive on that graph for the entire forward pass. With the v0.5.0 preset
  (20+ primitives, each producing canvas-sized intermediates at bs=20
  canvas=1024) this added **~10 GB VRAM** versus v0.4 — driving 24 GB peak
  on a 3090 where v0.4 ran at 14.7 GB. Reproduced on home Win bench as
  +5.25 GB on a 5060 Ti; the no_grad wrap drops it back to baseline.

### Compatibility

- **No API change.** If you were relying on aug being differentiable
  (extremely unusual — only adversarial-aug research), wrap the call in
  `torch.enable_grad()` to opt back in. We are not aware of any caller in
  this position.

## [0.5.1] - 2026-05-23

### Fixed

- **CI test failure on v0.5.0** — the new `test_background_compose_photo_dir`
  test in `tests/test_v050_ood.py` imported Pillow at function entry, but
  Pillow wasn't in the `[dev]` install-extras and CI runners install via
  `pip install -e ".[dev]"`. All four `python 3.9 / 3.10 / 3.11 / 3.12`
  CI matrices failed. Pillow is still an *optional* paraug runtime
  dependency (only needed when `background_compose.photo_dir` is set);
  the rest of paraug works without it.

### Changed

- New `paraug[photo]` install extra — `pip install paraug[photo]` pulls
  Pillow for callers that want to use the `photo_dir` background source.
- `[dev]` install-extra now also pulls Pillow so CI test runs cover the
  photo_dir code path on every push.
- `test_background_compose_photo_dir` now uses `pytest.importorskip("PIL.Image")`
  so the test skips gracefully (rather than fails) when Pillow isn't
  installed in a downstream user's test environment.

### Compatibility

- **No source change** from v0.5.0. v0.5.0 PyPI install works unchanged
  for callers that don't run the dev test suite. Behaviour at runtime is
  byte-identical; this is a CI-and-packaging fix only.

## [0.5.0] - 2026-05-23

### Added

- **`paraug.place_into_canvas(foreground, mask, canvas_size, fill, ...)`**
  — layout helper that embeds a foreground (and its mask) at a sampled
  position inside a larger constant-colour canvas, with random per-axis
  margins. Designed for nested-rectangle segmentation tasks (e.g. an ECG
  content rectangle inside a larger paper sheet) where the model must
  learn that the wider surrounding frame is a distractor it should
  ignore. Without random placement at training time, the model never
  sees content-fills-the-frame layout variation and over-segments to the
  outer rectangle at inference. Pair with `AugPipeline.compose` for the
  full layered-synthesis flow.
- **4 new photometric primitives** targeting the photo-realism gap that
  caused indoor-iPhone OOD failure (lab observation: synthetic IoU
  saturates at 0.999 while real-photo IoU was < 0.5):
  - `spatial_color_cast` — low-frequency 2D additive RGB shift map (vs
    `hue_shift`'s single-angle whole-frame rotation). Models non-uniform
    indoor-lighting tint that varies across the frame.
  - `paper_glare` — large elliptical bright reflection covering 10-30%
    of the frame, with adjustable aspect ratio. Differs from
    `specular_streaks` (thin line shapes) and `specular_highlight`
    (point highlights) — models a broad glossy-paper wash from angled
    ceiling lights.
  - `white_balance_shift` — per-channel gain along the Planckian
    blackbody locus (3000K-8000K). Constrained to the 1D WB curve so it
    looks "real-camera tinted" rather than `color_jitter`'s arbitrary
    3D per-channel jitter.
  - `defocus_blur` — spatially-varying gaussian blur (one or more
    in-focus blobs over a blurred whole-frame copy). Models phone
    autofocus error where one region is sharp and another is OOF.
- **`background_compose.photo_dir`** — `background_compose` now accepts
  a directory of real photos (jpg/jpeg/png/bmp/webp) in addition to the
  existing DTD `.npy` cache and procedural noise. Photos are scanned
  once and decoded lazily on first use, cached per-process. New
  `p_photo` spec key controls how often the photo source is chosen
  (priority order: photo > dtd > procedural).
- **`paraug.presets.OOD_PRINTED_PAPER`** — curated config dict that
  combines the new primitives at strengths tuned for the
  "printed-paper-photographed-by-phone-indoors" deployment. Use
  directly with `AugPipeline(presets.OOD_PRINTED_PAPER(), canvas_size=...)`
  or deepcopy + mutate for fine-tuning.
- `tests/test_v050_ood.py` — 17 tests covering layout, the 4 new
  primitives (including warm/cool WB asymmetry, defocus gradient
  energy, p=0 pass-through), `photo_dir` integration, preset
  copy-on-call semantics, and CPU/CUDA parity for the new primitives.

### Changed

- `paraug` now optionally imports Pillow at runtime when
  `background_compose.photo_dir` is used. Pillow is **not** a hard
  dependency — the rest of paraug works without it.

### Compatibility

- **No breaking changes.** Every v0.4.0 caller continues to work.
  `place_into_canvas` is opt-in, the new primitives are opt-in via
  config keys, the new `photo_dir` spec key on `background_compose`
  defaults to None (legacy DTD-or-procedural behaviour). Test suite
  grew 111 → 128, all passing.

## [0.4.0] - 2026-05-22

### Added

- **`AugPipeline.compose(foreground, background, mask)`** — blend a
  foreground onto a background through a mask, then run the configured
  aug. Data flow: (1) geometric primitives warp `(foreground, mask)`
  together while the background stays static, (2) blend
  `composite = fg_w * mask_w + background * (1 - mask_w)`, (3) photometric
  primitives perturb the composite, (4) optional `canvas_size` stretch.
  Accepts numpy `(H, W, C)` uint8/float arrays *or* torch tensors;
  output type matches the foreground input type. Building block for
  layered synthesis — e.g. "content printed on paper, then paper
  photographed in a scene" is two `compose` calls (pass-1 output becomes
  pass-2 foreground).
- **`canvas_size=(H, W)`** kwarg on `AugPipeline.__init__` — conforms
  every `__call__` / `compose` output to a fixed size via non-uniform
  `F.interpolate` stretch (input aspect ratio is NOT preserved; this is
  intentional for downstream uniform batching of variable-size inputs).
  Default `None` keeps output size equal to input size.
- `compose(..., return_transform=True)` returns a `transform` dict
  (`canvas_size`, `scale_x`, `scale_y`) so callers whose ground truth is
  stored as coordinates outside the tensor can rescale it consistently
  with the canvas stretch. (GT carried inside the tensor — a mask, or
  channels stacked via `n_image_channels` — rides the resize for free.)
- `tests/test_compose.py` — 15 tests covering blend correctness (numpy
  and tensor I/O), geometric/photometric application, canvas stretch on
  both `__call__` and `compose`, the transform dict, layered two-call
  synthesis (with and without canvas_size), background/mask auto-resize
  to the foreground frame, validation errors, and CPU/CUDA parity.
- `examples/04_compose_layered.py` — runnable two-pass layered-synthesis
  demo.

### Changed

- `__call__`'s geometric and photometric loops are factored into private
  `_apply_geometric` / `_apply_photometric` methods, shared with
  `compose`. No behaviour change for existing `__call__` callers.

### Compatibility

- **No breaking changes.** `compose` and `canvas_size` are additive;
  every existing `AugPipeline(config)` / `aug(img, mask, ...)` call site
  behaves identically. Test suite grew 96 → 111, all passing.

## [0.3.1] - 2026-05-17

### Changed

- `tests/test_pixel_position_consistency.py` now encodes BOTH source x and
  source y in the mask (via packed cell-id `x + y·W`) and asserts the
  img/mask grid agreement on both axes. The v0.3.0 test only validated
  source x, which would have silently missed a y-axis grid mismatch
  between img-bilinear and mask-nearest under a regression.
- `tests/test_n_image_channels.py::test_extra_channels_geometric_warped_same_grid`
  now encodes pixel POSITION per channel (x in even channels, y in odd)
  instead of a per-channel constant. A constant can't distinguish a shared
  grid from a per-channel-divergent grid (both leave constants unchanged);
  position-encoded values shift differently under different grids, so
  divergence is detectable.

### Added

- `tests/test_n_image_channels.py::test_photometric_returns_same_mask_object`
  — pins the photometric-primitive contract that `mask` is passed through
  by object identity, not just by value. Future contributors who replace
  `mask` with a new tensor (even one equal to the input) fail CI before
  mask-aware downstream pipelines see corrupted semantics.

### Documentation

- README / README_zh-TW: new "Sampling-mode note" subsection under
  "Stacking extra spatial channels" — extra channels are bilinear-sampled
  like img; `mask` is nearest-sampled. Discrete labels / class IDs must
  go through `mask`, not stacked extra channels.

### Fixes (downstream consumers — `ecg_aug.multi_channel_v2`)

These are external to paraug's published surface but are part of the
same 2026-05-17 review pass against the multi-channel GT path:

- **Edge-strip θ bias**: the per-strip slope extraction divided by a
  fixed `2·slope_step_px` denominator. At strips within `slope_step_px`
  pixels of an image edge, the ±-offset column samples clamped to the
  same boundary column — so the actual sample distance shrank to
  `slope_step_px` (or 0 at the very edge), but the slope was still
  divided by the original `2·slope_step_px`, yielding a wrong-by-factor
  estimate (or 0 at the corner where the distance collapsed entirely).
  Fixed: divide slope by the per-strip ACTUAL span (`x_hi − x_lo`); mark
  `thmask=0` where the span collapses to 0 so the downstream loss
  doesn't supervise on a phantom θ.
- **Phantom line "tails"**: rendering a short line segment used a clamped
  `frac` so the line's endpoint y was held constant outside the segment's
  x-range. The Gaussian then rendered a continuous ridge stretching across
  the whole image at that y. Fixed: out-of-range positions are pushed to
  +inf so the Gaussian decays to zero.
- Removed the unused `peak_thresh` kwarg from `extract_per_strip_gt_v2`.
- Added explicit guards: `H < 3` / `W < 3` and `slope_step_px < 1` now
  raise `ValueError` instead of silently breaking the sub-pixel parabolic
  refinement at index 0.

## [0.3.0] - 2026-05-17

### Added

- **Multi-channel `AugPipeline` input** — `AugPipeline(config, n_image_channels=N)`
  opts the call into channel-wise role-splitting. The first `N` leading
  channels are treated as "image" (geometric + photometric), the remaining
  channels are treated as extra spatial data that follows the geometric
  warp **but skips photometric perturbations**. Use case: stack GT
  heatmap/distance/tangent fields onto the image tensor so they share the
  back-warp grid for free, eliminating the need for a separate forward-warp
  solver downstream. `random_shadow` is correctly re-classified as
  photometric for the split (its dispatch is geometric but its effect is
  multiplicative, so extra channels see only true geometry).
- `tests/test_n_image_channels.py` — 7 unit tests covering: default
  back-compat (None) for any channel count, explicit `n_image_channels`
  split, photometric skip on extra channels, `random_shadow` re-routing,
  and validation on invalid values.
- `tests/test_pixel_position_consistency.py` — 6 guardrail tests verifying
  paraug's internal grid is shared between `img` and `mask` (img bilinear
  vs mask nearest stay within the 0.5 px quantisation bound). Useful for
  catching regressions in the geometric primitives' grid wiring.

### Changed

- `AugPipeline.__init__` gains `n_image_channels: int | None = None`. Default
  `None` preserves prior behaviour for any input channel count (no split).
  Explicit `int` enables the split.

### Compatibility

- **No breaking changes**: every existing call site continues to work
  unchanged. The new kwarg is opt-in via explicit `n_image_channels=`. Test
  suite grew from 88 to 95 tests, all passing.

## [0.1.2] - 2026-05-15

### Added

- `benchmarks/paraug_self.py` + `benchmarks/README.md` — self-benchmark
  measuring median latency and throughput across CPU and CUDA on a
  representative augmentation pipeline (affine + tps + gamma +
  gaussian_blur + gaussian_noise). Initial reference table from an NVIDIA
  RTX 5060 Ti is included; community PRs of additional hardware results
  welcome.

### Changed

- CI workflows bumped to `actions/checkout@v5` and enable
  `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` at workflow level to opt all
  Node-based actions into the Node 24 runtime ahead of GitHub's
  2026-06-02 default flip. Silences the deprecation warning seen in v0.1.1
  Actions runs.

## [0.1.1] - 2026-05-15

### Added

- README hero banner (`docs/banner.png`) and project logo (`docs/logo.png`).
- Traditional Chinese translation: `README_zh-TW.md` with a language switcher
  in the main README.
- `.github/workflows/publish.yml` — OIDC trusted-publisher workflow that
  builds an sdist + wheel and uploads to PyPI when a `v*` git tag is pushed.

### Changed

- Installation instructions now point to `pip install git+https://...` as the
  primary path until the PyPI release is live, instead of the previous
  unconditional `pip install paraug` (which would have failed).

## [0.1.0] - 2026-05-15

Initial public release.

### Added

- `AugPipeline(config)` — single entry point accepting a dict of geometric
  and photometric primitive specs. Returns `(image_aug, mask_aug)` with
  deterministic per-item RNG.
- **7 geometric primitives**: `affine`, `perspective`, `random_crop_pad`,
  `elastic_transform`, `optical_distortion`, `random_shadow`, `tps`.
- **24 photometric primitives**: gamma, gaussian_noise, gaussian_blur,
  motion_blur, salt_pepper_noise, color_jitter, hue_shift, random_grayscale,
  lighting, jpeg_approx, sharpness, clahe, local_contrast, vignette,
  specular_highlight, specular_streaks, cutout, salt_patches,
  paper_texture_overlay, watermark, random_text_overlay, background_compose,
  stains, creases.
- **CPU/GPU bit-exact parity**: per-primitive RNG is sampled on CPU
  regardless of tensor device, so the same seed produces identical output
  across CPU and CUDA back-ends within 1e-6 elementwise / 2e-4
  grid_sample-class tolerance.
- `set_deterministic(state=True, warn_only=True)` global toggle wrapping
  `torch.use_deterministic_algorithms` and cuDNN flags.
- `per_item_seed(seed_base, epoch, step, item_i, primitive)` and
  `cpu_generator(seed)` exposed for callers that need to mirror the RNG
  outside the pipeline.
- `scripts/audit_spec_keys.py` — AST-based check that every test fixture and
  example dict uses spec keys that match the primitive's
  `spec.get("KEY", default)` reads. Runs in CI; exits non-zero on mismatch.
