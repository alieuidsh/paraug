# Changelog

All notable changes to paraug are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
