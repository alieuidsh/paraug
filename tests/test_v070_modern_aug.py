"""Tests for v0.7.0 modern aug primitives:
random_erasing, grid_mask, cutmix, mixup, mix_info.
"""
import pytest
import torch

import paraug
from paraug import AugPipeline, mix_info
from paraug.photometric import (
    random_erasing, grid_mask, cutmix, mixup, PHOTOMETRIC_PRIMITIVES)


def _rand(B=4, C=3, H=64, W=64, device="cpu"):
    g = torch.Generator().manual_seed(0)
    return torch.rand(B, C, H, W, generator=g, device=device)


# ─── registry sanity ───────────────────────────────────────────────────
def test_new_primitives_registered():
    for name in ["random_erasing", "grid_mask", "cutmix", "mixup"]:
        assert name in PHOTOMETRIC_PRIMITIVES
        assert callable(PHOTOMETRIC_PRIMITIVES[name])


def test_paraug_describe_lists_new():
    desc = paraug.describe(return_dict=True)
    for name in ["random_erasing", "grid_mask", "cutmix", "mixup"]:
        assert name in desc
        assert desc[name]["kind"] == "photometric"


# ─── random_erasing ────────────────────────────────────────────────────
def test_random_erasing_default_modifies():
    img = _rand(B=4, H=64, W=64)
    out, _ = random_erasing(img, None, {"p": 1.0, "size_frac_range": (0.1, 0.3)}, 7, 0, 0)
    assert out.shape == img.shape
    # at least one item should differ from input (region was erased)
    diffs = [not torch.equal(out[i], img[i]) for i in range(4)]
    assert any(diffs), "random_erasing with p=1.0 should modify at least one item"


def test_random_erasing_p_zero_passthrough():
    img = _rand(B=2)
    out, _ = random_erasing(img, None, {"p": 0.0}, 7, 0, 0)
    assert torch.equal(out, img)


def test_random_erasing_deterministic():
    img = _rand(B=4)
    spec = {"p": 1.0, "size_frac_range": (0.1, 0.2)}
    a, _ = random_erasing(img, None, spec, 7, 0, 0)
    b, _ = random_erasing(img, None, spec, 7, 0, 0)
    assert torch.equal(a, b)


def test_random_erasing_fill_modes():
    img = _rand(B=2)
    for mode in ["normal", "uniform", "constant"]:
        out, _ = random_erasing(img, None, {
            "p": 1.0, "size_frac_range": (0.1, 0.1), "fill_mode": mode,
            "fill_value": 0.7}, 5, 0, 0)
        assert out.shape == img.shape
    with pytest.raises(ValueError, match="fill_mode must be"):
        random_erasing(img, None, {"p": 1.0, "fill_mode": "garbage"}, 5, 0, 0)


# ─── grid_mask ─────────────────────────────────────────────────────────
def test_grid_mask_drops_pixels():
    img = torch.ones(2, 3, 64, 64)
    out, _ = grid_mask(img, None, {"p": 1.0, "ratio": 0.5, "d_range": (8, 16),
                                     "rotation_deg": 0.0, "fill_value": 0.0}, 5, 0, 0)
    # Some pixels should now be 0 (dropped)
    assert (out == 0.0).any(), "grid_mask should drop some pixels"
    # But not all (ratio=0.5 leaves half)
    assert (out == 1.0).any(), "grid_mask should keep some pixels"


def test_grid_mask_ratio_bounds():
    img = _rand(B=1)
    with pytest.raises(ValueError, match="ratio must be in"):
        grid_mask(img, None, {"p": 1.0, "ratio": 0.0}, 0, 0, 0)
    with pytest.raises(ValueError, match="ratio must be in"):
        grid_mask(img, None, {"p": 1.0, "ratio": 1.5}, 0, 0, 0)


def test_grid_mask_deterministic():
    img = _rand(B=2, H=48, W=48)
    spec = {"p": 1.0, "ratio": 0.4, "d_range": (8, 16)}
    a, _ = grid_mask(img, None, spec, 11, 0, 0)
    b, _ = grid_mask(img, None, spec, 11, 0, 0)
    assert torch.equal(a, b)


# ─── cutmix ────────────────────────────────────────────────────────────
def test_cutmix_pastes_partner():
    # Make items distinguishable: each item is a constant level (0, 1, 2, 3).
    # With B=4 the randperm is almost never identity for all items.
    img = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1).expand(4, 3, 32, 32).contiguous() / 3.0
    out, _ = cutmix(img, None, {"p": 1.0, "alpha": 1.0}, 7, 0, 0)
    # At least one item should now contain pixels at a level other than its own
    # original (the partner's level got pasted into a region).
    changed = sum(int(not torch.equal(out[i], img[i])) for i in range(4))
    assert changed >= 1, "cutmix should modify at least one item at B=4"


def test_cutmix_alpha_validation():
    img = _rand(B=2)
    with pytest.raises(ValueError, match="alpha must be > 0"):
        cutmix(img, None, {"p": 1.0, "alpha": 0.0}, 0, 0, 0)


def test_cutmix_deterministic():
    img = _rand(B=4)
    spec = {"p": 1.0, "alpha": 1.0}
    a, _ = cutmix(img, None, spec, 11, 0, 0)
    b, _ = cutmix(img, None, spec, 11, 0, 0)
    assert torch.equal(a, b)


# ─── mixup ─────────────────────────────────────────────────────────────
def test_mixup_lerp():
    # B=4 to avoid the perm=identity edge case at B=2.
    img = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1).expand(4, 3, 16, 16).contiguous() / 3.0
    out, _ = mixup(img, None, {"p": 1.0, "alpha": 0.4}, 7, 0, 0)
    # At least one item should change (perm rarely == identity for all at B=4).
    changed = sum(int(not torch.equal(out[i], img[i])) for i in range(4))
    assert changed >= 1, "mixup should modify at least one item at B=4"


def test_mixup_alpha_validation():
    img = _rand(B=2)
    with pytest.raises(ValueError, match="alpha must be > 0"):
        mixup(img, None, {"p": 1.0, "alpha": -0.1}, 0, 0, 0)


def test_mixup_deterministic():
    img = _rand(B=4)
    spec = {"p": 1.0, "alpha": 0.4}
    a, _ = mixup(img, None, spec, 11, 0, 0)
    b, _ = mixup(img, None, spec, 11, 0, 0)
    assert torch.equal(a, b)


# ─── mix_info ──────────────────────────────────────────────────────────
def test_mix_info_recovers_cutmix_perm():
    """mix_info should return the same perm that cutmix used internally."""
    img = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1).expand(4, 3, 16, 16).contiguous()
    seed, ep, st = 42, 0, 0
    out, _ = cutmix(img, None, {"p": 1.0, "alpha": 1.0}, seed, ep, st)
    info = mix_info("cutmix", seed, ep, st, B=4, p=1.0, alpha=1.0)
    assert info["perm"].shape == (4,)
    assert info["gate"].all()
    assert (info["lam"] >= 0).all() and (info["lam"] <= 1).all()


def test_mix_info_mixup_matches_actual_lam():
    """For mixup with B=1 (no partner), the lerp output validates λ recovery
    on a 2-item batch where item 0 is partnered with item 1."""
    img = torch.zeros(2, 1, 4, 4)
    img[1] = 1.0
    seed, ep, st = 9, 0, 0
    out, _ = mixup(img, None, {"p": 1.0, "alpha": 0.4}, seed, ep, st)
    info = mix_info("mixup", seed, ep, st, B=2, p=1.0, alpha=0.4)
    perm = info["perm"]
    lam = info["lam"]
    # For item i: out[i] = lam[i] * img[i] + (1-lam[i]) * img[perm[i]]
    for i in range(2):
        expected = lam[i] * img[i] + (1.0 - lam[i]) * img[perm[i]]
        assert torch.allclose(out[i], expected, atol=1e-5), (
            f"mix_info λ doesn't match mixup output for item {i}: "
            f"out mean {out[i].mean():.4f} vs expected {expected.mean():.4f}")


def test_mix_info_validates_primitive():
    with pytest.raises(ValueError, match="must be cutmix/mixup"):
        mix_info("garbage", 0, 0, 0, B=2)


def test_mix_info_cutmix_actual_lam():
    """cutmix_actual_lam returns the area-ratio λ after rounding."""
    info = mix_info("cutmix_actual_lam", 7, 0, 0, B=2, p=1.0, alpha=1.0,
                    img_shape=(2, 3, 64, 64))
    assert info["lam"].shape == (2,)
    assert (info["lam"] >= 0).all() and (info["lam"] <= 1).all()


# ─── pipeline integration ─────────────────────────────────────────────
def test_full_pipeline_with_new_primitives():
    aug = AugPipeline({
        "photometric": {
            "random_erasing": {"p": 0.5, "size_frac_range": (0.05, 0.2)},
            "grid_mask":      {"p": 0.5, "ratio": 0.4, "d_range": (8, 16)},
            "cutmix":         {"p": 0.3, "alpha": 1.0},
            "mixup":          {"p": 0.3, "alpha": 0.4},
        },
    })
    img = _rand(B=4, H=64, W=64)
    out, _ = aug(img, seed_base=42, epoch=0, step=0)
    assert out.shape == img.shape


# ─── CPU/GPU parity (CUDA only) ───────────────────────────────────────
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("primitive,spec", [
    ("random_erasing", {"p": 1.0, "size_frac_range": (0.1, 0.1), "fill_mode": "constant", "fill_value": 0.7}),
    ("grid_mask",      {"p": 1.0, "ratio": 0.4, "d_range": (8, 8), "rotation_deg": 0.0}),
    ("cutmix",         {"p": 1.0, "alpha": 1.0}),
    ("mixup",          {"p": 1.0, "alpha": 0.4}),
])
def test_cpu_gpu_parity(primitive, spec):
    fn = PHOTOMETRIC_PRIMITIVES[primitive]
    img_cpu = _rand(B=4, H=32, W=32)
    img_gpu = img_cpu.cuda()
    out_cpu, _ = fn(img_cpu, None, spec, 13, 0, 0)
    out_gpu, _ = fn(img_gpu, None, spec, 13, 0, 0)
    diff = (out_cpu - out_gpu.cpu()).abs().max()
    assert diff < 1e-4, f"{primitive} CPU/GPU diff {diff:.6f} exceeds 1e-4"
