"""Tests for paraug v0.5.0 — OOD photo-realism additions.

Covers:
  - `place_into_canvas` helper (layout aug)
  - `spatial_color_cast`, `paper_glare`, `white_balance_shift`,
    `defocus_blur` photometric primitives
  - `background_compose.photo_dir` real-photo source extension
  - `presets.OOD_PRINTED_PAPER` preset config
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from paraug import AugPipeline, place_into_canvas, presets
from paraug.photometric import PHOTOMETRIC_PRIMITIVES


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 64, 80


# ─── place_into_canvas ────────────────────────────────────────────────
def test_place_into_canvas_numpy_basic():
    """Foreground placed at random position, output sized to canvas, mask
    matches the foreground footprint."""
    fg = np.full((H, W, 3), 100, dtype=np.uint8); fg[..., 0] = 255  # red fg
    mask = np.full((H, W), 255, dtype=np.uint8)
    out, out_mask = place_into_canvas(fg, mask, canvas_size=(200, 240),
                                        fill=(245, 245, 245),
                                        margin_frac_range=(0.10, 0.20),
                                        seed_base=0)
    assert isinstance(out, np.ndarray) and out.dtype == np.uint8
    assert out.shape == (200, 240, 3)
    assert out_mask.shape == (200, 240)
    # Mask must contain a contiguous foreground region (sum > 0)
    assert out_mask.sum() > 0, "mask should contain placed fg footprint"
    # Mask sum should be strictly less than the canvas area (margin exists)
    assert out_mask.sum() < 200 * 240 * 255
    # Background (mask==0) should be the fill colour
    bg_pix = (out_mask == 0)
    assert bg_pix.any()
    assert (out[bg_pix][..., 0] > 240).all(), "bg region should be fill colour"


def test_place_into_canvas_tensor_batch():
    """Tensor batch input → tensor output preserved shape."""
    fg = torch.zeros(3, 3, H, W, device=DEVICE)
    mask = torch.ones(3, 1, H, W, device=DEVICE)
    out, out_mask = place_into_canvas(fg, mask, canvas_size=(128, 160),
                                        fill=1.0, seed_base=42)
    assert torch.is_tensor(out)
    assert out.shape == (3, 3, 128, 160)
    assert out_mask.shape == (3, 1, 128, 160)


def test_place_into_canvas_center_deterministic():
    """position='center' ignores margin_frac_range and centres the fg."""
    fg = torch.full((1, 3, 50, 60), 0.5, device=DEVICE)
    mask = torch.ones(1, 1, 50, 60, device=DEVICE)
    out, out_mask = place_into_canvas(fg, mask, canvas_size=(100, 120),
                                        fill=0.0, position="center", seed_base=1)
    # With center placement: fg is at top=(100-50)//2=25, left=(120-60)//2=30
    assert out_mask[0, 0, 0, 0] == 0.0   # corner is bg fill
    assert out_mask[0, 0, 50, 60] == 1.0  # centre is fg


def test_place_into_canvas_seed_determinism():
    """Same seed → identical placement."""
    fg = torch.zeros(1, 3, 32, 40)
    mask = torch.ones(1, 1, 32, 40)
    a, am = place_into_canvas(fg, mask, (80, 100), fill=1.0, seed_base=7)
    b, bm = place_into_canvas(fg, mask, (80, 100), fill=1.0, seed_base=7)
    assert torch.equal(a, b) and torch.equal(am, bm)


def test_place_into_canvas_downscales_when_fg_too_big():
    """Foreground larger than canvas → down-fit (never upscale)."""
    fg = torch.zeros(1, 3, 300, 400)
    mask = torch.ones(1, 1, 300, 400)
    out, out_mask = place_into_canvas(fg, mask, canvas_size=(100, 100),
                                        fill=0.0, position="center", seed_base=0)
    assert out.shape == (1, 3, 100, 100)
    assert out_mask.shape == (1, 1, 100, 100)


def test_place_into_canvas_validation():
    fg = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)
    with pytest.raises(ValueError, match="position must be"):
        place_into_canvas(fg, mask, (100, 100), position="bogus")
    with pytest.raises(ValueError, match="canvas_size must be"):
        place_into_canvas(fg, mask, (0, 100))
    with pytest.raises(ValueError, match="margin_frac_range"):
        place_into_canvas(fg, mask, (100, 100), margin_frac_range=(0.6, 0.7))


# ─── new photometric primitives ───────────────────────────────────────
def test_spatial_color_cast_changes_image():
    img = torch.full((2, 3, H, W), 0.5, device=DEVICE)
    out, _ = PHOTOMETRIC_PRIMITIVES["spatial_color_cast"](
        img, None, {"p": 1.0, "amplitude": 0.15, "sigma": 4.0},
        seed_base=0, epoch=0, step=0)
    assert (out - 0.5).abs().max() > 0.01, "should add a non-trivial cast"


def test_spatial_color_cast_pass_through_non_rgb():
    img = torch.full((1, 1, H, W), 0.5, device=DEVICE)  # 1-channel
    out, _ = PHOTOMETRIC_PRIMITIVES["spatial_color_cast"](
        img, None, {"p": 1.0}, 0, 0, 0)
    assert torch.equal(out, img), "non-RGB should pass through"


def test_paper_glare_brightens_region():
    img = torch.full((1, 3, H, W), 0.4, device=DEVICE)
    out, _ = PHOTOMETRIC_PRIMITIVES["paper_glare"](
        img, None,
        {"p": 1.0, "intensity_range": (0.5, 0.5),
         "area_frac_range": (0.30, 0.30), "softness": 1.0},
        seed_base=0, epoch=0, step=0)
    assert out.max() > 0.8, "glare should add a noticeably bright region"
    assert (out >= img).all(), "glare adds brightness, never darkens"


def test_white_balance_shift_warm_vs_cool():
    """A warm WB (3500K → R gain up, B gain down) shifts image toward red;
    a cool WB (8000K) shifts toward blue."""
    img = torch.full((1, 3, H, W), 0.5, device=DEVICE)
    warm, _ = PHOTOMETRIC_PRIMITIVES["white_balance_shift"](
        img, None, {"p": 1.0, "temp_range_K": (3500.0, 3500.0)}, 0, 0, 0)
    cool, _ = PHOTOMETRIC_PRIMITIVES["white_balance_shift"](
        img, None, {"p": 1.0, "temp_range_K": (8000.0, 8000.0)}, 0, 0, 0)
    # warm: R > B; cool: B > R
    assert warm[0, 0].mean() > warm[0, 2].mean(), "3500K should look warm"
    assert cool[0, 2].mean() > cool[0, 0].mean(), "8000K should look cool"


def test_white_balance_shift_pass_through_non_rgb():
    img = torch.full((1, 1, H, W), 0.5, device=DEVICE)
    out, _ = PHOTOMETRIC_PRIMITIVES["white_balance_shift"](
        img, None, {"p": 1.0}, 0, 0, 0)
    assert torch.equal(out, img)


def test_defocus_blur_softens_non_focus_region():
    """A noisy input has sharp gradients; defocus should reduce gradient
    energy somewhere (the out-of-focus region)."""
    g = torch.Generator().manual_seed(0)
    img = torch.rand(1, 3, H, W, generator=g, device="cpu").to(DEVICE)
    out, _ = PHOTOMETRIC_PRIMITIVES["defocus_blur"](
        img, None,
        {"p": 1.0, "sigma_range": (3.0, 3.0), "focus_frac_range": (0.1, 0.1),
         "n_focus_regions": 1},
        seed_base=0, epoch=0, step=0)
    grad_in = (img[..., 1:] - img[..., :-1]).abs().sum().item()
    grad_out = (out[..., 1:] - out[..., :-1]).abs().sum().item()
    assert grad_out < grad_in * 0.9, "defocus should reduce some gradient energy"


def test_new_primitives_p_zero_passthrough():
    """p=0 → bit-identical pass-through (same convention as other primitives)."""
    img = torch.rand(2, 3, H, W, device=DEVICE)
    for name in ["spatial_color_cast", "paper_glare", "white_balance_shift"]:
        out, _ = PHOTOMETRIC_PRIMITIVES[name](
            img.clone(), None, {"p": 0.0}, 0, 0, 0)
        assert torch.equal(out, img), f"{name} p=0 should pass-through"


# ─── background_compose photo_dir ──────────────────────────────────────
def test_background_compose_photo_dir(tmp_path):
    """photo_dir produces a real-photo background when p_photo > 0."""
    from PIL import Image
    # Plant 2 distinct photos
    arr1 = np.full((40, 50, 3), 60, dtype=np.uint8); arr1[..., 1] = 200  # green
    arr2 = np.full((40, 50, 3), 60, dtype=np.uint8); arr2[..., 2] = 200  # blue
    Image.fromarray(arr1).save(tmp_path / "photo1.jpg")
    Image.fromarray(arr2).save(tmp_path / "photo2.png")

    img = torch.full((4, 3, H, W), 0.5, device=DEVICE)
    mask = torch.zeros(4, 1, H, W, device=DEVICE)
    mask[:, :, : H // 2, : W // 2] = 1.0   # quarter is foreground

    spec = {
        "p": 1.0,
        "p_photo": 1.0,     # always pull from photo_dir
        "p_dtd": 0.0,
        "photo_dir": str(tmp_path),
        "channel_jitter": 0.0,
        "bg_brightness_range": (1.0, 1.0),
    }
    out, _ = PHOTOMETRIC_PRIMITIVES["background_compose"](
        img, mask, spec, seed_base=42, epoch=0, step=0)
    # Foreground quarter unchanged at 0.5; bg region should be one of the photos
    bg_region = out[:, :, H // 2:, W // 2:]   # bottom-right quadrant = bg only
    # Each item's bg should be RGB-dominated by one of our photos (G or B)
    mean_per_item_per_ch = bg_region.mean(dim=(2, 3))   # (4, 3)
    g_or_b_dominant = ((mean_per_item_per_ch[:, 1] > mean_per_item_per_ch[:, 0])
                       | (mean_per_item_per_ch[:, 2] > mean_per_item_per_ch[:, 0]))
    assert g_or_b_dominant.any(), "at least one item should have a photo bg"


# ─── presets ───────────────────────────────────────────────────────────
def test_preset_ood_printed_paper_returns_independent_copy():
    a = presets.OOD_PRINTED_PAPER()
    b = presets.OOD_PRINTED_PAPER()
    a["geometric"]["affine"] = {"p": 1.0}     # mutate a
    assert "affine" not in b["geometric"], "presets must return deep copies"


def test_preset_ood_printed_paper_builds_pipeline():
    """The preset config builds without KeyError and runs forward."""
    cfg = presets.OOD_PRINTED_PAPER()
    # Remove background_compose since we have no real photo dir handy
    cfg["photometric"].pop("background_compose", None)
    aug = AugPipeline(cfg, canvas_size=(96, 128))
    fg = np.full((H, W, 3), 200, dtype=np.uint8)
    bg = np.full((H, W, 3), 80, dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8); mask[10:-10, 10:-10] = 255
    img, m = aug.compose(fg, bg, mask, seed_base=0)
    assert img.shape == (96, 128, 3)


# ─── CPU/GPU parity for the new primitives ────────────────────────────
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_new_primitives_cpu_cuda_parity():
    """Per-item RNG sampled on CPU → bit-exact within ulp class on CUDA."""
    img_cpu = torch.rand(2, 3, H, W, generator=torch.Generator().manual_seed(0))
    img_cu = img_cpu.cuda()
    for name in ["spatial_color_cast", "paper_glare", "white_balance_shift"]:
        out_cpu, _ = PHOTOMETRIC_PRIMITIVES[name](img_cpu, None, {"p": 1.0}, 5, 0, 0)
        out_cu, _ = PHOTOMETRIC_PRIMITIVES[name](img_cu, None, {"p": 1.0}, 5, 0, 0)
        diff = (out_cpu - out_cu.cpu()).abs().max().item()
        assert diff < 2e-4, f"{name}: CPU vs CUDA max_diff={diff:.2e}"
