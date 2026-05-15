"""AugPipeline — single entry point for paraug.

Run a configured chain of geometric + photometric primitives on an image
(plus optional mask) with CPU/GPU bit-exact parity.

Usage:

    from paraug import AugPipeline

    aug = AugPipeline({
        "geometric": {"affine": {"p": 1.0, "rot_deg": 15}},
        "photometric": {"gamma": {"p": 0.5}},
    })
    img_out, mask_out = aug(img, mask=mask, seed_base=42)

`seed_base / epoch / step` together pin per-item RNG so the same arguments
produce bit-exact results across CPU and CUDA back-ends (see
`tests/test_parity.py`).
"""
from typing import Dict, Optional, Tuple

import torch

from .geometric import GEOMETRIC_PRIMITIVES
from .photometric import PHOTOMETRIC_PRIMITIVES


class AugPipeline:
    """Run a chain of geometric and photometric primitives in declaration order.

    Args:
        config: dict with optional keys "geometric" and "photometric", each a
            dict mapping primitive name → spec dict. Primitives run in dict
            insertion order. Unknown primitive names raise KeyError at __init__.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.geom_specs = []
        for name, spec in config.get("geometric", {}).items():
            if name not in GEOMETRIC_PRIMITIVES:
                raise KeyError(f"unknown geometric primitive: {name!r}; "
                               f"available: {sorted(GEOMETRIC_PRIMITIVES)}")
            self.geom_specs.append((name, spec))
        self.photo_specs = []
        for name, spec in config.get("photometric", {}).items():
            if name in ("pick_n", "primitives"):
                continue
            if name not in PHOTOMETRIC_PRIMITIVES:
                raise KeyError(f"unknown photometric primitive: {name!r}; "
                               f"available: {sorted(PHOTOMETRIC_PRIMITIVES)}")
            self.photo_specs.append((name, spec))

    def __call__(self,
                 img: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 seed_base: int = 0,
                 epoch: int = 0,
                 step: int = 0,
                 ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the pipeline on (img, mask).

        Args:
            img: (B, C, H, W) float tensor.
            mask: optional (B, 1, H, W) float tensor warped alongside the image.
            seed_base: base seed for per-item RNG. Combine with `epoch` and
                `step` so the same dataset position gets different aug across
                epochs while staying deterministic per (seed_base, epoch, step).
            epoch, step: additional RNG axes (see `utils.per_item_seed`).

        Returns:
            (img_out, mask_out) — same shapes as inputs. `mask_out` is None
            iff `mask` is None.
        """
        for name, spec in self.geom_specs:
            img, mask = GEOMETRIC_PRIMITIVES[name](img, mask, spec,
                                                   seed_base, epoch, step)
        for name, spec in self.photo_specs:
            img, mask = PHOTOMETRIC_PRIMITIVES[name](img, mask, spec,
                                                     seed_base, epoch, step)
        return img, mask
