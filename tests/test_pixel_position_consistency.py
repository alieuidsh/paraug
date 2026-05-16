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
    mask: 1-channel encoding (x + y·W) / (W·H − 1) so a single nearest sample
    can be decoded back into both source x and source y. Lets the test catch
    grid mismatches on either axis (x-only encoding would silently miss a
    y-axis grid bug, since the y-coordinate of the source isn't recorded
    in the mask)."""
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=DTYPE),
        torch.arange(W, dtype=DTYPE),
        indexing="ij",
    )
    r = xx / (W - 1)
    g = yy / (H - 1)
    b = torch.ones_like(r)
    img = torch.stack([r, g, b], dim=0).unsqueeze(0)
    # mask values: integer cell id = x + y·W, in [0, W·H − 1]. Normalised to
    # [0, 1] for float storage; decode by multiplying by (W·H − 1) and
    # splitting via // W and % W.
    cell_id = (xx + yy * W).to(DTYPE)
    mask = (cell_id / (W * H - 1)).unsqueeze(0).unsqueeze(0)
    return img.to(DEVICE), mask.to(DEVICE)


@pytest.mark.parametrize("primitive,spec", list(PRIMITIVE_SPECS.items()))
def test_img_mask_grid_sync(primitive, spec):
    """img bilinear and mask nearest must come from the SAME grid for BOTH
    axes. Compares both the source x-coordinate (img channel 0 bilinear vs
    mask cell-id nearest → x) and the source y-coordinate (img channel 1
    vs mask cell-id → y); each must agree to within the 0.5 px nearest
    quantisation bound."""
    img, mask = _encoded_img_and_mask()
    aug = AugPipeline({"geometric": {primitive: spec}})
    img_out, mask_out = aug(img, mask=mask, seed_base=SEED_BASE)

    valid = img_out[0, 2] > VALID_THRESH
    # Decode mask cell id → (sx, sy).
    mask_cell = (mask_out[0, 0] * (W * H - 1)).round()
    decoded_mask_x = mask_cell % W
    decoded_mask_y = mask_cell // W
    decoded_img_x = img_out[0, 0] * (W - 1)
    decoded_img_y = img_out[0, 1] * (H - 1)
    diff_x = (decoded_img_x - decoded_mask_x).abs()
    diff_y = (decoded_img_y - decoded_mask_y).abs()

    if not valid.any():
        pytest.skip(f"{primitive}: 0 valid pixels (entirely padded)")
    max_dx = float(diff_x[valid].max().item())
    max_dy = float(diff_y[valid].max().item())
    assert max_dx < MAX_DIFF_PX, (
        f"{primitive}: img/mask src-x disagree by max={max_dx:.3f}px — "
        f"x-axis grid mismatch between img bilinear and mask nearest.")
    assert max_dy < MAX_DIFF_PX, (
        f"{primitive}: img/mask src-y disagree by max={max_dy:.3f}px — "
        f"y-axis grid mismatch between img bilinear and mask nearest.")


if __name__ == "__main__":
    print(f"device={DEVICE}  H={H} W={W}  acceptance < {MAX_DIFF_PX} px (both axes)\n")
    header = f"{'primitive':<20} {'max_dx':>10} {'max_dy':>10} {'n_valid':>10} {'verdict':>8}"
    print(header)
    print("-" * len(header))
    for name, spec in PRIMITIVE_SPECS.items():
        img, mask = _encoded_img_and_mask()
        aug = AugPipeline({"geometric": {name: spec}})
        img_out, mask_out = aug(img, mask=mask, seed_base=SEED_BASE)
        valid = img_out[0, 2] > VALID_THRESH
        mask_cell = (mask_out[0, 0] * (W * H - 1)).round()
        mx = mask_cell % W; my = mask_cell // W
        dx = ((img_out[0, 0] * (W - 1) - mx).abs())[valid]
        dy = ((img_out[0, 1] * (H - 1) - my).abs())[valid]
        max_dx = float(dx.max().item()) if dx.numel() else 0.0
        max_dy = float(dy.max().item()) if dy.numel() else 0.0
        verdict = "PASS" if max(max_dx, max_dy) < MAX_DIFF_PX else "FAIL"
        print(f"{name:<20} {max_dx:>10.4f} {max_dy:>10.4f} "
              f"{int(valid.sum().item()):>10} {verdict:>8}")
