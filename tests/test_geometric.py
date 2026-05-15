"""Geometric primitive smoke tests."""
import pytest
import torch

from paraug import AugPipeline, GEOMETRIC_PRIMITIVES


GEOM_SPECS = {
    "affine":             {"p": 1.0, "rot_deg": 15.0,
                            "scale_range": (0.9, 1.1), "translate_frac": 0.05},
    "perspective":        {"p": 1.0, "max_disp_frac": 0.05},
    "random_crop_pad":    {"p": 1.0, "min_scale": 0.7},
    "elastic_transform":  {"p": 1.0, "alpha": 8.0},
    "optical_distortion": {"p": 1.0, "k": 0.2},
    "random_shadow":      {"p": 1.0, "strength": 0.4},
    "tps":                {"p": 1.0, "max_disp": 12.0, "n_ctrl": 5},
}


def _make_input(B=2, C=3, H=64, W=64, with_mask=True):
    g = torch.Generator(device="cpu").manual_seed(0)
    img = torch.rand(B, C, H, W, generator=g)
    mask = torch.ones(B, 1, H, W) if with_mask else None
    return img, mask


@pytest.mark.parametrize("name", list(GEOM_SPECS))
def test_geometric_smoke(name):
    """Each primitive runs without crashing and preserves shape."""
    img, mask = _make_input()
    aug = AugPipeline({"geometric": {name: GEOM_SPECS[name]}})
    img_out, mask_out = aug(img, mask=mask, seed_base=42, epoch=0, step=0)
    assert img_out.shape == img.shape, f"{name}: shape changed {img.shape} -> {img_out.shape}"
    assert mask_out.shape == mask.shape, f"{name}: mask shape changed"
    assert not torch.isnan(img_out).any(), f"{name}: NaN in img_out"


@pytest.mark.parametrize("name", list(GEOM_SPECS))
def test_geometric_identity_p_zero(name):
    """p=0 should pass-through bit-identical (gated off per-item)."""
    img, mask = _make_input()
    spec = {**GEOM_SPECS[name], "p": 0.0}
    aug = AugPipeline({"geometric": {name: spec}})
    img_out, mask_out = aug(img, mask=mask, seed_base=42, epoch=0, step=0)
    assert torch.equal(img_out, img), f"{name}: p=0 image changed"
    assert torch.equal(mask_out, mask), f"{name}: p=0 mask changed"


def test_geometric_no_mask():
    """Mask=None path returns mask_out=None."""
    img, _ = _make_input(with_mask=False)
    aug = AugPipeline({"geometric": {"affine": GEOM_SPECS["affine"]}})
    img_out, mask_out = aug(img, mask=None, seed_base=42)
    assert mask_out is None
    assert img_out.shape == img.shape


def test_geometric_unknown_raises():
    with pytest.raises(KeyError):
        AugPipeline({"geometric": {"not_a_real_primitive": {"p": 1.0}}})


def test_geometric_primitive_keys_match_dispatch():
    """GEOMETRIC_PRIMITIVES dispatch contains all 7 advertised primitives."""
    expected = {"affine", "perspective", "random_crop_pad",
                "elastic_transform", "optical_distortion",
                "random_shadow", "tps"}
    assert set(GEOMETRIC_PRIMITIVES) == expected
