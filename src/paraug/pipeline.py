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

Multi-channel input (n_image_channels)
======================================
By default ALL channels in the input tensor receive every primitive
(geometric + photometric) — same as before the 2026-05-16 multi-channel
extension. Set `n_image_channels=N` explicitly to opt in to the split:
the first N channels get geometric + photometric, the remaining channels
get geometric only. Use case: GT heatmap/tangent fields stacked with RGB
so they share the back-warp grid for free, eliminating the need for a
separate forward-warp solver. `random_shadow` is geometric in dispatch
but multiplicative in effect, so when the split is active it's treated
as photometric (extra channels see only true geometry).
"""
from typing import Dict, Optional, Tuple

import torch

from .geometric import GEOMETRIC_PRIMITIVES
from .photometric import PHOTOMETRIC_PRIMITIVES


# Primitives that live in GEOMETRIC_PRIMITIVES (no forward-warp counterpart)
# but whose effect is multiplicative/photometric. When extra channels are
# present they must NOT see these — otherwise GT heatmaps would inherit the
# image shadow factor.
_GEOMETRIC_AS_PHOTOMETRIC = {"random_shadow"}


class AugPipeline:
    """Run a chain of geometric and photometric primitives in declaration order.

    Args:
        config: dict with optional keys "geometric" and "photometric", each a
            dict mapping primitive name → spec dict. Primitives run in dict
            insertion order. Unknown primitive names raise KeyError at __init__.
        n_image_channels: optional. Number of leading channels that count as
            "image" (receive both geometric warp and photometric
            perturbations). Channels beyond N follow geometric warp only.
            Default None = treat ALL channels as image (no split) — keeps
            behaviour identical to pre-2026-05-16 paraug for every channel
            count. Set explicitly to an int when stacking extra channels
            (e.g. GT heatmaps) that must skip photometric.
    """

    def __init__(self, config: Dict, n_image_channels: Optional[int] = None):
        self.config = config
        self.n_image_channels = n_image_channels
        if n_image_channels is not None and n_image_channels < 1:
            raise ValueError(
                f"n_image_channels must be >=1 or None; got {n_image_channels}")
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
            img: (B, C, H, W) float tensor. C >= n_image_channels. Channels
                [0, n_image_channels) get geometric + photometric aug; channels
                [n_image_channels, C) get geometric only.
            mask: optional (B, 1, H, W) float tensor warped alongside the image.
            seed_base: base seed for per-item RNG. Combine with `epoch` and
                `step` so the same dataset position gets different aug across
                epochs while staying deterministic per (seed_base, epoch, step).
            epoch, step: additional RNG axes (see `utils.per_item_seed`).

        Returns:
            (img_out, mask_out) — same shapes as inputs. `mask_out` is None
            iff `mask` is None.
        """
        # None → treat every channel as image (no split, original behaviour).
        n = self.n_image_channels if self.n_image_channels is not None else img.shape[1]
        has_extra = img.shape[1] > n
        if img.shape[1] < n:
            raise ValueError(
                f"img has {img.shape[1]} channels but n_image_channels={n}")

        for name, spec in self.geom_specs:
            if has_extra and name in _GEOMETRIC_AS_PHOTOMETRIC:
                img_main = img[:, :n].contiguous()
                img_extra = img[:, n:]
                img_main, mask = GEOMETRIC_PRIMITIVES[name](
                    img_main, mask, spec, seed_base, epoch, step)
                img = torch.cat([img_main, img_extra], dim=1)
            else:
                img, mask = GEOMETRIC_PRIMITIVES[name](
                    img, mask, spec, seed_base, epoch, step)

        if self.photo_specs:
            if has_extra:
                img_main = img[:, :n].contiguous()
                img_extra = img[:, n:]
                for name, spec in self.photo_specs:
                    img_main, mask = PHOTOMETRIC_PRIMITIVES[name](
                        img_main, mask, spec, seed_base, epoch, step)
                img = torch.cat([img_main, img_extra], dim=1)
            else:
                for name, spec in self.photo_specs:
                    img, mask = PHOTOMETRIC_PRIMITIVES[name](
                        img, mask, spec, seed_base, epoch, step)
        return img, mask
