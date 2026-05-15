# Changelog

All notable changes to paraug are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
