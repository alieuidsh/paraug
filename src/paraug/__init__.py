"""paraug — Parity Augmentation library for image + mask.

Bit-exact CPU/GPU parity across 7 geometric + 24 photometric primitives.
Single entry point: build with `AugPipeline(config)`, then call with
`aug(img, mask=None, seed_base=..., epoch=..., step=...)` returning
`(img_out, mask_out)`. Per-item RNG is sampled on CPU regardless of tensor
device, so the same seed produces identical output on CPU and CUDA back-ends
within a tight tolerance (1e-6 elementwise, 2e-4 for grid_sample-class ops).

Quickstart:

    from paraug import AugPipeline
    aug = AugPipeline({"geometric": {"affine": {"p": 1.0, "rot_deg": 15}}})
    img_out, mask_out = aug(img, mask=mask, seed_base=42)
"""
from .pipeline import AugPipeline
from .utils import set_deterministic, per_item_seed, cpu_generator
from .geometric import GEOMETRIC_PRIMITIVES
from .photometric import PHOTOMETRIC_PRIMITIVES
from .layout import place_into_canvas
from . import presets

__version__ = "0.5.2"

__all__ = [
    "AugPipeline",
    "set_deterministic",
    "per_item_seed",
    "cpu_generator",
    "GEOMETRIC_PRIMITIVES",
    "PHOTOMETRIC_PRIMITIVES",
    "place_into_canvas",
    "presets",
    "__version__",
]
