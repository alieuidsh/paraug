"""Tests for AugPipeline n_image_channels API.

Verify the multi-channel split:
- Geometric primitives apply to ALL channels (same grid_sample).
- Photometric primitives apply ONLY to first n_image_channels.
- random_shadow (geometric in dispatch, multiplicative in effect) follows
  the photometric rule when extra channels are present.
- Default n_image_channels=3 + 3-channel input is bit-identical to legacy.
"""
import pytest
import torch

from paraug import AugPipeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 64, 64


def _make_input(C, B=2):
    g = torch.Generator(device="cpu").manual_seed(0)
    img = torch.rand(B, C, H, W, generator=g).to(DEVICE)
    return img


def test_n_image_channels_default_backcompat_3ch():
    """Default n_image_channels=None + 3ch input matches legacy behaviour."""
    img = _make_input(3)
    cfg = {
        "geometric": {"affine": {"p": 1.0, "rot_deg": 10.0}},
        "photometric": {"gamma": {"p": 1.0, "gamma_range": (0.8, 1.2)}},
    }
    aug_default = AugPipeline(cfg)                       # None → no split
    aug_explicit = AugPipeline(cfg, n_image_channels=3)  # 3 of 3 → no extra
    out_d, _ = aug_default(img.clone(), seed_base=42)
    out_e, _ = aug_explicit(img.clone(), seed_base=42)
    assert torch.equal(out_d, out_e), "default None vs explicit 3 diverge for 3ch input"


def test_n_image_channels_default_backcompat_9ch():
    """Default n_image_channels=None + 9ch input applies photometric to ALL
    channels — preserving the pre-2026-05-16 behaviour that just augmented
    every channel uniformly. Setting n_image_channels=3 is opt-in to skip."""
    img = _make_input(9)
    img_orig = img.clone()
    cfg = {"photometric": {"gamma": {"p": 1.0, "gamma_range": (0.5, 0.5)}}}
    aug_default = AugPipeline(cfg)  # None → no split → photometric ALL ch
    out, _ = aug_default(img.clone(), seed_base=42)
    # Every channel must differ from input (gamma applied uniformly).
    for c in range(9):
        assert not torch.equal(out[:, c], img_orig[:, c]), (
            f"default None: channel {c} unchanged by photometric — back-compat broken")


def test_extra_channels_geometric_warped_same_grid():
    """9ch input under affine: all 9 channels get the same back-warp grid."""
    img = _make_input(9)
    # Encode a unique constant per channel so we can verify the grid was shared.
    for c in range(9):
        img[:, c] = float(c)
    cfg = {"geometric": {"affine": {"p": 1.0, "rot_deg": 15.0,
                                      "scale_range": (0.9, 1.1)}}}
    aug = AugPipeline(cfg, n_image_channels=3)
    out, _ = aug(img, seed_base=42)
    # After warp, each output pixel sampled the same src pixel for all channels.
    # The constant-per-channel encoding means each channel still has its own
    # constant value at any in-image pixel (modulo padding-zero leaks). Check
    # that interior pixels still hold each channel's distinct constant.
    interior = out[:, :, 10:-10, 10:-10]
    for c in range(9):
        ch_vals = interior[:, c]
        # All interior pixels of ch=c should equal float(c) (the back-warp
        # of a constant is still that constant).
        assert (ch_vals - float(c)).abs().max() < 1e-4, (
            f"channel {c} not constant after geometric warp — grid not shared "
            f"or photometric leaked")


def test_extra_channels_photometric_skip():
    """9ch input + gamma photometric: channels 3-8 must be unchanged."""
    img = _make_input(9)
    img_orig = img.clone()
    cfg = {"photometric": {"gamma": {"p": 1.0, "gamma_range": (0.5, 0.5)}}}
    aug = AugPipeline(cfg, n_image_channels=3)
    out, _ = aug(img, seed_base=42)
    # Channels 3-8 must be bit-identical to input
    assert torch.equal(out[:, 3:], img_orig[:, 3:]), (
        "photometric leaked into extra channels (3-8)")
    # Channels 0-2 must be DIFFERENT (gamma applied)
    assert not torch.equal(out[:, :3], img_orig[:, :3]), (
        "photometric did not apply to image channels (0-2)")


def test_extra_channels_random_shadow_skip():
    """random_shadow is geometric in dispatch but multiplicative — with extra
    channels, it must be treated as photometric (skip extras)."""
    img = _make_input(9)
    # Constant 0.5 across all channels so we can detect shadow modulation.
    img[:] = 0.5
    img_orig = img.clone()
    cfg = {"geometric": {"random_shadow": {"p": 1.0, "strength": 0.4,
                                             "softness_px": 5.0}}}
    aug = AugPipeline(cfg, n_image_channels=3)
    out, _ = aug(img, seed_base=42)
    # Channels 3-8 unchanged
    assert torch.equal(out[:, 3:], img_orig[:, 3:]), (
        "random_shadow leaked into extra channels (3-8) — should be treated "
        "as photometric for multi-channel input")
    # Channels 0-2 are darkened somewhere
    assert (out[:, :3] - img_orig[:, :3]).abs().max() > 0, (
        "random_shadow did not modulate image channels (0-2)")


def test_n_image_channels_invalid():
    img = _make_input(3)
    with pytest.raises(ValueError, match="n_image_channels must be"):
        AugPipeline({}, n_image_channels=0)
    with pytest.raises(ValueError, match="n_image_channels=5"):
        AugPipeline({}, n_image_channels=5)(img)


def test_n_image_channels_extra_channels_default_no_split():
    """6-channel input with default n_image_channels=None: random_shadow is
    treated as geometric (its dispatch class) — applied to ALL channels, no
    split. Confirms the opt-in nature of the split."""
    img = _make_input(6)
    img[:] = 0.5
    img_orig = img.clone()
    cfg = {"geometric": {"random_shadow": {"p": 1.0, "strength": 0.4,
                                             "softness_px": 5.0}}}
    aug = AugPipeline(cfg)  # default None → no split
    out, _ = aug(img, seed_base=42)
    # All 6 channels should see shadow modulation somewhere
    for c in range(6):
        assert (out[:, c] - img_orig[:, c]).abs().max() > 0, (
            f"default None: random_shadow did not modulate channel {c} — "
            f"split was wrongly active")
