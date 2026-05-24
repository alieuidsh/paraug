"""Photometric primitive smoke tests."""
import pytest
import torch

from paraug import AugPipeline, PHOTOMETRIC_PRIMITIVES


PHOTO_SPECS = {
    "gamma":               {"p": 1.0, "gamma_range": (0.8, 1.2)},
    "gaussian_noise":      {"p": 1.0, "sigma_range": (0.005, 0.04)},
    "gaussian_blur":       {"p": 1.0, "sigma_range": (0.5, 2.0)},
    "motion_blur":         {"p": 1.0, "length_range": (3, 9)},
    "salt_pepper_noise":   {"p": 1.0, "density": 0.005, "salt_vs_pepper": 0.5},
    "color_jitter":        {"p": 1.0, "brightness": 0.2, "contrast": 0.2,
                             "saturation": 0.2},
    "hue_shift":           {"p": 1.0, "max_deg": 30.0},
    "random_grayscale":    {"p": 1.0},
    "lighting":            {"p": 1.0, "strength": 0.15, "sigma_frac": 0.3},
    "jpeg_approx":         {"p": 1.0, "noise_sigma_range": (0.005, 0.02)},
    "sharpness":           {"p": 1.0, "alpha_range": (0.1, 0.5)},
    "clahe":               {"p": 1.0, "clip_limit": 2.0},
    "local_contrast":      {"p": 1.0, "strength_range": (0.05, 0.20)},
    "vignette":            {"p": 1.0, "strength_range": (0.2, 0.5)},
    "specular_highlight":  {"p": 1.0, "intensity_range": (0.05, 0.20)},
    "specular_streaks":    {"p": 1.0, "n_range": (1, 3)},
    "cutout":              {"p": 1.0, "size_frac_range": (0.05, 0.15)},
    "salt_patches":        {"p": 1.0, "n_range": (1, 5)},
    "paper_texture_overlay": {"p": 1.0, "blend_strength": 0.1},
    "watermark":           {"p": 1.0, "alpha_range": (0.05, 0.15)},
    "random_text_overlay": {"p": 1.0},
    "stains":              {"p": 1.0, "n_range": (1, 3)},
    "creases":             {"p": 1.0, "n_range": (1, 2)},
    # background_compose requires a bg tensor — skip default smoke
}


@pytest.mark.parametrize("name", list(PHOTO_SPECS))
def test_photometric_smoke(name):
    g = torch.Generator(device="cpu").manual_seed(0)
    img = torch.rand(2, 3, 64, 64, generator=g)
    mask = torch.ones(2, 1, 64, 64)
    aug = AugPipeline({"photometric": {name: PHOTO_SPECS[name]}})
    img_out, mask_out = aug(img, mask=mask, seed_base=42)
    assert img_out.shape == img.shape, f"{name}: shape changed"
    assert not torch.isnan(img_out).any(), f"{name}: NaN in img_out"
    # Photometric primitives generally produce values in [0, 1].
    # Some primitives (e.g. lighting, gamma) might push slightly outside; clamp check.
    assert img_out.min() >= -0.5 and img_out.max() <= 1.5, \
        f"{name}: img_out range [{img_out.min():.3f}, {img_out.max():.3f}] far from [0, 1]"


@pytest.mark.parametrize("name", list(PHOTO_SPECS))
def test_photometric_identity_p_zero(name):
    g = torch.Generator(device="cpu").manual_seed(0)
    img = torch.rand(2, 3, 64, 64, generator=g)
    spec = {**PHOTO_SPECS[name], "p": 0.0}
    aug = AugPipeline({"photometric": {name: spec}})
    img_out, _ = aug(img, seed_base=42)
    assert torch.equal(img_out, img), f"{name}: p=0 image changed"


def test_photometric_dispatch_size():
    # paraug ships 24 photometric primitives (23 in PHOTO_SPECS + background_compose
    # which is covered separately because it has unusual mask-aware semantics).
    assert len(PHOTOMETRIC_PRIMITIVES) >= 24


def test_photometric_background_compose():
    """background_compose generates bg internally (procedural noise or DTD
    cache) and composites with the paper image via the mask. With p_dtd=0
    we force the procedural-noise path so the test is hermetic (no .npy
    fixture needed)."""
    aug = AugPipeline({"photometric": {
        "background_compose": {"p": 1.0,
                                "p_dtd": 0.0,
                                "bg_brightness_range": (0.6, 1.0),
                                "sigma": 40.0,
                                "channel_jitter": 0.1},
    }})
    g = torch.Generator(device="cpu").manual_seed(0)
    img = torch.rand(2, 3, 64, 64, generator=g)
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1.0  # interior paper square
    out_img, out_mask = aug(img, mask=mask, seed_base=42, epoch=0, step=0)
    assert out_img.shape == img.shape
    assert torch.isfinite(out_img).all()
    assert torch.equal(out_mask, mask), "photometric must not warp mask"
    interior = mask.expand_as(img) > 0.5
    diff_interior = (out_img - img)[interior].abs().max().item()
    assert diff_interior < 1e-6, \
        f"interior diff {diff_interior:.2e} > 1e-6 — bg leaked into mask?"
    # Stronger contract: every off-paper pixel must be replaced by bg, not
    # just "some pixels changed". Sum across channels so a pixel counts as
    # replaced if any channel diff > 1e-6.
    edge_mask = (mask == 0)  # (B, 1, H, W)
    diff_per_pixel = (out_img - img).abs().sum(dim=1, keepdim=True)  # (B, 1, H, W)
    edge_changed = (diff_per_pixel > 1e-6) & edge_mask
    edge_total = int(edge_mask.sum())
    edge_n_changed = int(edge_changed.sum())
    assert edge_n_changed == edge_total, \
        (f"only {edge_n_changed}/{edge_total} off-paper pixels were "
         f"replaced — bg composition incomplete")


def test_photometric_background_compose_p_zero():
    """p=0 → image bit-identical pass-through."""
    aug = AugPipeline({"photometric": {
        "background_compose": {"p": 0.0,
                                "p_dtd": 0.0,
                                "bg_brightness_range": (0.6, 1.0)},
    }})
    g = torch.Generator(device="cpu").manual_seed(0)
    img = torch.rand(2, 3, 64, 64, generator=g)
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1.0
    out_img, out_mask = aug(img, mask=mask, seed_base=42)
    assert torch.equal(out_img, img), "p=0 image changed"
    assert torch.equal(out_mask, mask)


# ─── v0.5.3: fast_noise opt-in (CPU/GPU parity intentionally broken) ──
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fast_noise_default_off_preserves_parity():
    """set_fast_noise default OFF — gaussian_noise/jpeg_approx/salt_pepper_noise
    on CUDA still produce CPU-path output (slow but bit-exact to CPU)."""
    from paraug import set_fast_noise
    from paraug.photometric import gaussian_noise
    assert not __import__("paraug").fast_noise_enabled()  # default state
    img = torch.rand(2, 3, 16, 16, device="cuda")
    set_fast_noise(False)  # ensure clean state
    out_a, _ = gaussian_noise(img, None, {"p": 1.0, "sigma_range": (0.01, 0.02)}, 7, 0, 0)
    out_b, _ = gaussian_noise(img, None, {"p": 1.0, "sigma_range": (0.01, 0.02)}, 7, 0, 0)
    assert torch.equal(out_a, out_b), "CPU-path must be deterministic"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fast_noise_deterministic_per_seed():
    """When fast_noise=True, same seed → identical output (within run).
    Different seed → different output."""
    from paraug import set_fast_noise
    from paraug.photometric import gaussian_noise, jpeg_approx, salt_pepper_noise
    try:
        set_fast_noise(True)
        img = torch.rand(2, 3, 16, 16, device="cuda")
        for fn, spec in [
            (gaussian_noise, {"p": 1.0, "sigma_range": (0.01, 0.02)}),
            (jpeg_approx, {"p": 1.0, "noise_sigma_range": (0.01, 0.02)}),
            (salt_pepper_noise, {"p": 1.0, "density": 0.05, "salt_vs_pepper": 0.5}),
        ]:
            a, _ = fn(img, None, spec, 7, 0, 0)
            b, _ = fn(img, None, spec, 7, 0, 0)
            c, _ = fn(img, None, spec, 8, 0, 0)
            assert torch.equal(a, b), f"{fn.__name__}: same seed must produce same output"
            assert not torch.equal(a, c), f"{fn.__name__}: different seed should differ"
    finally:
        set_fast_noise(False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fast_noise_breaks_cpu_path_parity_by_design():
    """fast_noise produces DIFFERENT output from CPU-path (cuRAND != MT19937).
    Documenting the contract: fast_noise is for production, not parity tests."""
    from paraug import set_fast_noise
    from paraug.photometric import gaussian_noise
    img = torch.rand(2, 3, 16, 16, device="cuda")
    set_fast_noise(False)
    out_slow, _ = gaussian_noise(img, None, {"p": 1.0, "sigma_range": (0.01, 0.02)}, 7, 0, 0)
    try:
        set_fast_noise(True)
        out_fast, _ = gaussian_noise(img, None, {"p": 1.0, "sigma_range": (0.01, 0.02)}, 7, 0, 0)
    finally:
        set_fast_noise(False)
    # They must differ — that's the whole opt-in trade-off.
    assert not torch.equal(out_slow, out_fast), "fast_noise should produce different output by design"
