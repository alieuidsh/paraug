"""CPU/GPU bit-exact parity tests.

paraug's defining property: the same `seed_base / epoch / step` produces the
same output across CPU and CUDA back-ends within a tight tolerance:

- Elementwise ops (gamma, noise, color jitter, …): atol 1e-6
- grid_sample-class ops (affine, perspective, tps, …): atol 2e-4

If CUDA isn't available these tests skip gracefully so the suite still passes
on CPU-only CI runners.
"""
import pytest
import torch


cuda_available = torch.cuda.is_available()


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

# Split photometric primitives by which tolerance they should meet.
# Truly elementwise ops (mul/add/where, no conv) hit 1e-6 ulp drift.
# Anything that calls F.conv2d / F.avg_pool / F.interpolate hits cuDNN paths
# and gets bumped to grid_sample-class tolerance.
PHOTO_ELEMENTWISE = {
    "gamma":          {"p": 1.0, "gamma_range": (0.8, 1.2)},
    "gaussian_noise": {"p": 1.0, "sigma_range": (0.005, 0.04)},
    "color_jitter":   {"p": 1.0, "brightness": 0.2, "contrast": 0.2,
                        "saturation": 0.2},
    "vignette":       {"p": 1.0, "strength_range": (0.2, 0.5)},
    "cutout":         {"p": 1.0, "size_frac_range": (0.05, 0.15)},
    "hue_shift":      {"p": 1.0, "max_deg": 30.0},
}

PHOTO_CONV = {
    "gaussian_blur":  {"p": 1.0, "sigma_range": (0.5, 2.0)},
    "motion_blur":    {"p": 1.0, "length_range": (3, 9)},
}


TOL_ELEMENTWISE = 1e-6
TOL_GRID_SAMPLE = 2e-4


def _make_input(device, B=2, C=3, H=64, W=64):
    g = torch.Generator(device="cpu").manual_seed(0)
    img = torch.rand(B, C, H, W, generator=g)
    mask = torch.ones(B, 1, H, W)
    return img.to(device), mask.to(device)


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
@pytest.mark.parametrize("name", list(GEOM_SPECS))
def test_parity_geometric_cpu_vs_cuda(name):
    """Each geometric primitive: CPU and CUDA outputs match within grid-sample tolerance."""
    from paraug import AugPipeline
    aug = AugPipeline({"geometric": {name: GEOM_SPECS[name]}})
    img_cpu, mask_cpu = _make_input("cpu")
    img_cuda, mask_cuda = _make_input("cuda")
    out_cpu = aug(img_cpu, mask=mask_cpu, seed_base=42, epoch=0, step=0)
    out_cuda = aug(img_cuda, mask=mask_cuda, seed_base=42, epoch=0, step=0)
    img_diff = (out_cpu[0] - out_cuda[0].cpu()).abs().max().item()
    mask_diff = (out_cpu[1] - out_cuda[1].cpu()).abs().max().item()
    assert img_diff < TOL_GRID_SAMPLE, \
        f"{name}: img CPU vs CUDA max_diff={img_diff:.2e} > {TOL_GRID_SAMPLE:.0e}"
    assert mask_diff < TOL_GRID_SAMPLE, \
        f"{name}: mask CPU vs CUDA max_diff={mask_diff:.2e} > {TOL_GRID_SAMPLE:.0e}"


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
@pytest.mark.parametrize("name", list(PHOTO_ELEMENTWISE))
def test_parity_photometric_elementwise(name):
    """Pure elementwise photometric ops: CPU vs CUDA within 1e-6 tolerance.

    This is the core 1e-6 claim from README. Any primitive in this list must
    hold the tight tolerance — if it doesn't, either fix the implementation
    or move it to PHOTO_CONV.
    """
    from paraug import AugPipeline
    aug = AugPipeline({"photometric": {name: PHOTO_ELEMENTWISE[name]}})
    img_cpu, _ = _make_input("cpu")
    img_cuda, _ = _make_input("cuda")
    out_cpu = aug(img_cpu, seed_base=42)
    out_cuda = aug(img_cuda, seed_base=42)
    diff = (out_cpu[0] - out_cuda[0].cpu()).abs().max().item()
    assert diff < TOL_ELEMENTWISE, \
        f"{name}: CPU vs CUDA max_diff={diff:.2e} > {TOL_ELEMENTWISE:.0e}"


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
@pytest.mark.parametrize("name", list(PHOTO_CONV))
def test_parity_photometric_conv(name):
    """Conv-class photometric ops (motion_blur, gaussian_blur): CPU vs CUDA
    within grid_sample-class tolerance. cuDNN convolutions diverge from ATen
    conv at the few-ulp level."""
    from paraug import AugPipeline
    aug = AugPipeline({"photometric": {name: PHOTO_CONV[name]}})
    img_cpu, _ = _make_input("cpu")
    img_cuda, _ = _make_input("cuda")
    out_cpu = aug(img_cpu, seed_base=42)
    out_cuda = aug(img_cuda, seed_base=42)
    diff = (out_cpu[0] - out_cuda[0].cpu()).abs().max().item()
    assert diff < TOL_GRID_SAMPLE, \
        f"{name}: CPU vs CUDA max_diff={diff:.2e} > {TOL_GRID_SAMPLE:.0e}"


def test_deterministic_repeat_cpu():
    """Same seed → same output across two calls (CPU-only, runs on all CI)."""
    from paraug import AugPipeline
    aug = AugPipeline({
        "geometric": {"affine": GEOM_SPECS["affine"]},
        "photometric": {"gamma": PHOTO_ELEMENTWISE["gamma"]},
    })
    img, mask = _make_input("cpu")
    out1 = aug(img.clone(), mask=mask.clone(), seed_base=42)
    out2 = aug(img.clone(), mask=mask.clone(), seed_base=42)
    assert torch.equal(out1[0], out2[0]), "image not deterministic on repeat"
    assert torch.equal(out1[1], out2[1]), "mask not deterministic on repeat"
