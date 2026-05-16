"""Pixel-perfect img ↔ mask grid-sync test (Suho 2026-05-16 spec, scope-A).

This test verifies that inside one paraug primitive call, the SAME warp grid
is applied to both `img` and `mask`. We encode the source pixel coordinate
into both channels then check that, for each output pixel sampled from inside
the image, the bilinear-sampled img decodes to (within nearest-quantisation
tolerance) the same source position that the nearest-sampled mask reports.

Scope: tests paraug only. It does NOT exercise downstream forward-warp paths
(e.g. ecg_aug's `warp_lines_gt_gpu`); those are covered by a separate
cross-path test in the heartbeat repo.

A failure here would mean paraug itself is sampling img and mask with
different grids — that has never been the suspected bug, so this test is
expected to pass and acts as a guardrail.

Tolerance:
  - img uses bilinear; on a linear position encoding, bilinear interp is
    EXACT (sub-ulp).
  - mask uses nearest; on a linear position encoding, nearest can be off by
    up to 0.5 px in each axis (the half-pixel snap of the nearest sample).

So |decoded(img) - decoded(mask)| <= 0.5 px is the strict bound. We use
0.51 px to allow for floating-point slack on grid quantisation.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from paraug import AugPipeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

H, W = 256, 256
SEED_BASE = 42

PRIMITIVE_SPECS = {
    "affine":             {"p": 1.0, "rot_deg": 15.0,
                            "scale_range": (0.9, 1.1), "translate_frac": 0.05},
    "perspective":        {"p": 1.0, "max_disp_frac": 0.05},
    "random_crop_pad":    {"p": 1.0, "min_scale": 0.7},
    "elastic_transform":  {"p": 1.0, "alpha": 8.0},
    "optical_distortion": {"p": 1.0, "k": 0.2},
    "tps":                {"p": 1.0, "max_disp": 18.0, "n_ctrl": 5},
}

# Strict bound: nearest can quantise off the bilinear pos by up to 0.5 px.
MAX_DIFF_PX = 0.51

# Validity filter strictness. The validity channel is bilinear of constant
# 1.0 inside; it equals 1.0 only when ALL 4 source neighbours of the
# back-warp are inside. Any neighbour outside drops it below 1.0 AND breaks
# the linear position encoding (padded 0 leaks into the bilinear of img[0]
# even though only one corner is missing). Require essentially-1 to ensure
# decoding is exact.
VALID_THRESH = 0.9999


def _encoded_img_and_mask():
    """img: 3-channel (R=x/(W-1), G=y/(H-1), B=1.0 validity).
    mask: 1-channel encoding x + y*(W) packed as float so we can decode it back.
    Use mask = x / (W - 1) (single axis); paraug masks are (B, 1, H, W).
    """
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=DTYPE),
        torch.arange(W, dtype=DTYPE),
        indexing="ij",
    )
    r = xx / (W - 1)
    g = yy / (H - 1)
    b = torch.ones_like(r)
    img = torch.stack([r, g, b], dim=0).unsqueeze(0)
    mask_x = (xx / (W - 1)).unsqueeze(0).unsqueeze(0)
    return img.to(DEVICE), mask_x.to(DEVICE)


@pytest.mark.parametrize("primitive,spec", list(PRIMITIVE_SPECS.items()))
def test_img_mask_grid_sync(primitive, spec):
    """img bilinear and mask nearest must come from the same grid."""
    img, mask = _encoded_img_and_mask()
    aug = AugPipeline({"geometric": {primitive: spec}})
    img_out, mask_out = aug(img, mask=mask, seed_base=SEED_BASE)

    # Both outputs encode the SRC x-coord (img channel 0, mask channel 0).
    # Restrict to pixels where the back-warp bilinear stayed fully inside src
    # (validity > VALID_THRESH); otherwise padded-zero contaminates the
    # linear encoding and the decoded x is artificially pulled toward 0.
    valid = img_out[0, 2] > VALID_THRESH
    decoded_img_x = img_out[0, 0] * (W - 1)
    decoded_mask_x = mask_out[0, 0] * (W - 1)
    diff = (decoded_img_x - decoded_mask_x).abs()

    diff_valid = diff[valid]
    if diff_valid.numel() == 0:
        pytest.skip(f"{primitive}: 0 valid pixels (entirely padded)")
    max_diff = float(diff_valid.max().item())
    mean_diff = float(diff_valid.mean().item())
    assert max_diff < MAX_DIFF_PX, (
        f"{primitive}: img/mask src-x disagree by max={max_diff:.3f}px "
        f"(mean={mean_diff:.3f}, n_valid={int(valid.sum().item())}). "
        f"This means img and mask were warped with different grids."
    )


if __name__ == "__main__":
    print(f"device={DEVICE}  H={H} W={W}  acceptance < {MAX_DIFF_PX} px\n")
    header = f"{'primitive':<20} {'max':>10} {'mean':>10} {'n_valid':>10} {'verdict':>8}"
    print(header)
    print("-" * len(header))
    for name, spec in PRIMITIVE_SPECS.items():
        img, mask = _encoded_img_and_mask()
        aug = AugPipeline({"geometric": {name: spec}})
        img_out, mask_out = aug(img, mask=mask, seed_base=SEED_BASE)
        valid = img_out[0, 2] > VALID_THRESH
        diff = ((img_out[0, 0] - mask_out[0, 0]).abs() * (W - 1))[valid]
        max_d = float(diff.max().item()) if diff.numel() else 0.0
        mean_d = float(diff.mean().item()) if diff.numel() else 0.0
        verdict = "PASS" if max_d < MAX_DIFF_PX else "FAIL"
        print(f"{name:<20} {max_d:>10.4f} {mean_d:>10.4f} "
              f"{int(valid.sum().item()):>10} {verdict:>8}")
