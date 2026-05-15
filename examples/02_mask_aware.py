"""02 — mask-aware geometric augmentation.

Geometric primitives warp the image and the mask together with the same
transform, so a segmentation mask stays aligned with the image.

Before running, install paraug (e.g. `pip install -e .` from repo root or
`PYTHONPATH=src python ...`), then:
    python examples/02_mask_aware.py
"""
import torch

from paraug import AugPipeline


def main():
    B, H, W = 2, 64, 64
    g = torch.Generator(device="cpu").manual_seed(0)
    img = torch.rand(B, 3, H, W, generator=g)
    # Toy mask: a centred 32x32 square of 1s on a 0 background.
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, H // 4: 3 * H // 4, W // 4: 3 * W // 4] = 1.0

    aug = AugPipeline({
        "geometric": {
            "perspective":      {"p": 1.0, "max_disp_frac": 0.10},
            "random_crop_pad":  {"p": 1.0, "min_scale": 0.7},
        },
    })

    img_out, mask_out = aug(img, mask=mask, seed_base=7)

    # Sanity check: mask after aug is still {0, 1}-ish (nearest sampling).
    unique = mask_out.unique()
    print(f"mask unique values after warp: {unique.tolist()}")
    assert set(unique.tolist()).issubset({0.0, 1.0}), \
        "mask should stay binary under nearest sampling"
    print(f"image shape {tuple(img_out.shape)} / mask shape {tuple(mask_out.shape)}")
    print(f"mask area before: {int(mask.sum())} / after: {int(mask_out.sum())}")


if __name__ == "__main__":
    main()
