"""01 — quickstart: minimal load → augment → save round-trip.

Before running, install paraug (one of):
    pip install paraug                            # once on PyPI
    pip install -e .                              # editable install from repo root
    PYTHONPATH=src python examples/01_quickstart.py  # ad-hoc, no install

Then:
    python examples/01_quickstart.py
"""
import torch

from paraug import AugPipeline


def main():
    # Pretend we loaded a small batch of images.
    g = torch.Generator(device="cpu").manual_seed(0)
    img = torch.rand(2, 3, 128, 128, generator=g)
    print(f"input shape:  {tuple(img.shape)}, range [{img.min():.3f}, {img.max():.3f}]")

    aug = AugPipeline({
        "geometric": {
            "affine": {"p": 1.0, "rot_deg": 15.0,
                        "scale_range": (0.9, 1.1)},
        },
        "photometric": {
            "gamma":        {"p": 0.5, "gamma_range": (0.8, 1.2)},
            "gaussian_noise": {"p": 0.3, "sigma_range": (0.005, 0.02)},
        },
    })

    img_out, _ = aug(img, seed_base=42, epoch=0, step=0)
    print(f"output shape: {tuple(img_out.shape)}, range "
          f"[{img_out.min():.3f}, {img_out.max():.3f}]")

    # Determinism check: same call → same output.
    img_again, _ = aug(img.clone(), seed_base=42, epoch=0, step=0)
    assert torch.equal(img_out, img_again), "non-deterministic!"
    print("deterministic on repeat (same seed_base/epoch/step): OK")


if __name__ == "__main__":
    main()
