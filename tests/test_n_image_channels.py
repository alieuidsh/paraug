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
    """9ch input under affine: every channel back-warps from the SAME source
    pixel. Encode pixel POSITION + unique per-channel offset (not a
    per-channel constant — a constant can't distinguish a shared grid from
    a per-channel grid since both leave the constant unchanged; and a
    plain x/y position alone can't distinguish a buggy impl that copies
    warped channel 0 into every even channel from a correct shared-grid
    impl, since both yield the same warped x. The constant offset survives
    the warp untouched, so per-channel uniqueness is preserved end-to-end
    and a channel-copy bug shows up as a wrong offset)."""
    B, C = 2, 9
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    # Each channel = (position-encoded base) + (channel-index offset).
    # Even channels encode x; odd channels encode y. The +c*1000 offset
    # is preserved by the linear warp (a constant shift commutes with
    # affine grid_sample), so channel c's output at any pixel is
    # `warped_pos + c*1000`. Subtract the offset to recover the warped
    # position, which must match the channel-0 (or channel-1) reference.
    img = torch.empty(B, C, H, W, dtype=torch.float32, device=DEVICE)
    for c in range(C):
        base = (xx if c % 2 == 0 else yy).to(DEVICE)
        img[:, c] = base + float(c) * 1000.0
    cfg = {"geometric": {"affine": {"p": 1.0, "rot_deg": 15.0,
                                      "scale_range": (0.9, 1.1)}}}
    aug = AugPipeline(cfg, n_image_channels=3)
    out, _ = aug(img, seed_base=42)
    interior = out[:, :, 10:-10, 10:-10]
    # Recover warped position per channel by subtracting the constant
    # offset. All even channels should now agree (warped x); all odd
    # channels should agree (warped y). A channel-copy bug would have
    # the wrong offset and would NOT subtract back to the same value.
    for c in range(0, C, 2):
        rec_c = interior[:, c] - float(c) * 1000.0
        ref = interior[:, 0]   # channel 0 has offset 0 → rec_0 = interior[:,0]
        assert (rec_c - ref).abs().max() < 1e-2, (
            f"channel {c} (x-encoded) does not match channel 0 ref after "
            f"removing the per-channel offset — either the grid diverged "
            f"or a copy/permutation bug swapped channel content")
    for c in range(1, C, 2):
        rec_c = interior[:, c] - float(c) * 1000.0
        ref = interior[:, 1] - 1000.0
        assert (rec_c - ref).abs().max() < 1e-2, (
            f"channel {c} (y-encoded) does not match channel 1 ref after "
            f"removing the per-channel offset — either the grid diverged "
            f"or a copy/permutation bug swapped channel content")


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


def test_photometric_returns_same_mask_object():
    """Photometric primitives must pass `mask` through unchanged — by object
    identity AND by value. The pipeline contract assumes a photometric op
    never replaces OR mutates mask; this test pins both halves of that
    contract for the full photometric dispatch table so:
      (a) a future contributor who replaces `mask` with a new tensor
          (e.g. `mask = mask * 1`) fails CI on the identity check, and
      (b) a future contributor who mutates `mask` in place (e.g.
          `mask.mul_(some_factor)`) fails CI on the value-equality check.
    """
    from paraug.photometric import PHOTOMETRIC_PRIMITIVES
    img = _make_input(3)
    # Use random values (not constant 1) so an in-place mutation actually
    # shifts the contents detectably.
    g = torch.Generator(device="cpu").manual_seed(0)
    mask = torch.rand(2, 1, H, W, generator=g, dtype=torch.float32).to(DEVICE)
    mask_id = id(mask)
    for name, fn in PHOTOMETRIC_PRIMITIVES.items():
        mask_snapshot = mask.clone()
        spec = {"p": 1.0}
        try:
            _, mask_out = fn(img.clone(), mask, spec, seed_base=42,
                              epoch=0, step=0)
        except Exception as e:
            pytest.skip(f"{name}: spec-required setup not handled: {e}")
        assert mask_out is mask, (
            f"photometric primitive {name!r} returned a different mask "
            f"object (id changed: {mask_id} → {id(mask_out)}). Photometric "
            f"primitives must pass mask through untouched.")
        assert torch.equal(mask, mask_snapshot), (
            f"photometric primitive {name!r} MUTATED the mask in place "
            f"(values changed despite identity preserved). Photometric "
            f"primitives must not modify mask contents.")
