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

Fixed output size (canvas_size)
===============================
Set `canvas_size=(H, W)` so every output is conformed to a fixed size,
regardless of input size. The conform is a plain non-uniform stretch
(`F.interpolate`) — input aspect ratio is NOT preserved. This is the
right choice when downstream batching needs uniform shapes and the task
is consistent under stretch (e.g. grid detection: train and inference
both stretch to the same canvas, so the model learns in canvas space).
Default `None` keeps the output size equal to the input.

compose()
=========
`compose(foreground, background, mask)` blends a foreground onto a
background through a mask, then runs the configured aug. See the method
docstring for the full data flow. It is the building block for layered
synthesis (e.g. "ECG printed on paper, then paper photographed in a
scene" = two compose calls).
"""
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .geometric import GEOMETRIC_PRIMITIVES
from .photometric import PHOTOMETRIC_PRIMITIVES


# Primitives that live in GEOMETRIC_PRIMITIVES (no forward-warp counterpart)
# but whose effect is multiplicative/photometric. When extra channels are
# present they must NOT see these — otherwise GT heatmaps would inherit the
# image shadow factor.
_GEOMETRIC_AS_PHOTOMETRIC = {"random_shadow"}


def _is_numpy(x) -> bool:
    return isinstance(x, np.ndarray)


def _img_to_tensor(x: Union[np.ndarray, torch.Tensor]
                   ) -> Tuple[torch.Tensor, bool]:
    """Normalise an image input to a (B, C, H, W) float tensor in [0, 1].

    Accepts:
      - numpy (H, W, C) or (H, W) — uint8 [0,255] or float [0,1]
      - torch (B, C, H, W) or (C, H, W) float

    Returns (tensor, was_numpy_hwc). `was_numpy_hwc` records that the input
    came in as a single numpy HWC image so the output can be converted back.
    """
    if _is_numpy(x):
        t = torch.from_numpy(np.ascontiguousarray(x))
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            t = t.float()
        if t.dim() == 2:                 # (H, W) → (1, 1, H, W)
            t = t[None, None]
        elif t.dim() == 3:               # (H, W, C) → (1, C, H, W)
            t = t.permute(2, 0, 1)[None].contiguous()
        else:
            raise ValueError(
                f"numpy image must be (H,W) or (H,W,C); got shape {x.shape}")
        return t, True
    # torch tensor
    t = x
    if t.dim() == 3:                     # (C, H, W) → (1, C, H, W)
        t = t[None]
    elif t.dim() != 4:
        raise ValueError(
            f"torch image must be (C,H,W) or (B,C,H,W); got shape {tuple(t.shape)}")
    return t.float(), False


def _mask_to_tensor(x: Union[np.ndarray, torch.Tensor]
                    ) -> Tuple[torch.Tensor, bool]:
    """Normalise a mask input to a (B, 1, H, W) float tensor in [0, 1]."""
    if _is_numpy(x):
        t = torch.from_numpy(np.ascontiguousarray(x))
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            t = t.float()
        if t.dim() == 2:                 # (H, W) → (1, 1, H, W)
            t = t[None, None]
        elif t.dim() == 3:               # (H, W, 1) → (1, 1, H, W)
            t = t.permute(2, 0, 1)[None].contiguous()
        else:
            raise ValueError(
                f"numpy mask must be (H,W) or (H,W,1); got shape {x.shape}")
        return t, True
    t = x
    if t.dim() == 2:                     # (H, W) → (1, 1, H, W)
        t = t[None, None]
    elif t.dim() == 3:                   # (B, H, W) → (B, 1, H, W)
        t = t[:, None]
    elif t.dim() != 4:
        raise ValueError(
            f"torch mask must be (H,W)/(B,H,W)/(B,1,H,W); got {tuple(t.shape)}")
    return t.float(), False


def _tensor_to_img_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a (1, C, H, W) float tensor back to a numpy (H, W, C) uint8."""
    arr = t[0].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    return arr.permute(1, 2, 0).contiguous().cpu().numpy()


def _tensor_to_mask_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a (1, 1, H, W) float tensor back to a numpy (H, W) uint8."""
    arr = t[0, 0].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    return arr.contiguous().cpu().numpy()


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
        canvas_size: optional (H, W). When set, every output of `__call__` and
            `compose` is stretched (non-uniform `F.interpolate`) to this fixed
            size. Default None keeps output size equal to input size.
    """

    def __init__(self, config: Dict,
                 n_image_channels: Optional[int] = None,
                 canvas_size: Optional[Tuple[int, int]] = None):
        self.config = config
        self.n_image_channels = n_image_channels
        if n_image_channels is not None and n_image_channels < 1:
            raise ValueError(
                f"n_image_channels must be >=1 or None; got {n_image_channels}")
        if canvas_size is not None:
            if (not isinstance(canvas_size, (tuple, list))
                    or len(canvas_size) != 2
                    or int(canvas_size[0]) < 1 or int(canvas_size[1]) < 1):
                raise ValueError(
                    f"canvas_size must be (H, W) with H, W >= 1 or None; "
                    f"got {canvas_size!r}")
            canvas_size = (int(canvas_size[0]), int(canvas_size[1]))
        self.canvas_size = canvas_size
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

    # ─── core phases (shared by __call__ and compose) ──────────────────
    def _apply_geometric(self, img, mask, seed_base, epoch, step):
        """Run every geometric primitive on (img, mask). Honours the
        n_image_channels split: when extra channels are present, primitives
        in _GEOMETRIC_AS_PHOTOMETRIC are restricted to the first N channels."""
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
        return img, mask

    def _apply_photometric(self, img, mask, seed_base, epoch, step):
        """Run every photometric primitive on (img, mask). Honours the
        n_image_channels split: extra channels skip photometric entirely."""
        if not self.photo_specs:
            return img, mask
        n = self.n_image_channels if self.n_image_channels is not None else img.shape[1]
        has_extra = img.shape[1] > n
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

    def _resize_to_canvas(self, img, mask):
        """Stretch img (bilinear) and mask (nearest) to `self.canvas_size`.
        No-op when canvas_size is None or the size already matches."""
        if self.canvas_size is None:
            return img, mask
        H, W = self.canvas_size
        if img.shape[-2:] != (H, W):
            img = F.interpolate(img, size=(H, W), mode="bilinear",
                                align_corners=False)
        if mask is not None and mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode="nearest")
        return img, mask

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
            (img_out, mask_out). When `canvas_size` is None the shapes equal
            the inputs; when set, both are stretched to `canvas_size`.
            `mask_out` is None iff `mask` is None.
        """
        # Aug is preprocessing, never backpropped. Without no_grad, primitive
        # intermediates pile up on any live autograd graph the caller's inputs
        # carry — a training loop that forgets to detach can blow VRAM by GB.
        with torch.no_grad():
            img, mask = self._apply_geometric(img, mask, seed_base, epoch, step)
            img, mask = self._apply_photometric(img, mask, seed_base, epoch, step)
            img, mask = self._resize_to_canvas(img, mask)
        return img, mask

    def compose(self,
                foreground: Union[np.ndarray, torch.Tensor],
                background: Union[np.ndarray, torch.Tensor],
                mask: Union[np.ndarray, torch.Tensor],
                seed_base: int = 0,
                epoch: int = 0,
                step: int = 0,
                return_transform: bool = False):
        """Blend `foreground` onto `background` through `mask`, then augment.

        Data flow
        ---------
        1. **geometric** primitives warp (foreground, mask) together — the
           foreground sheet rotates / scales / warps while the background
           stays put (it is the static backdrop).
        2. **blend**: ``composite = fg_w * mask_w + background * (1 - mask_w)``
           — `mask_w` = 1 shows the warped foreground, 0 shows background.
        3. **photometric** primitives perturb the `composite` (lighting,
           colour, blur, …). The mask passes through photometric unchanged.
        4. **canvas_size** stretch, if the pipeline was built with one.

        Layered synthesis (e.g. "ECG printed on paper → paper photographed
        in a scene") is two `compose` calls: pass 1 output becomes pass 2's
        `foreground`. Use two `AugPipeline` instances if the two passes need
        different aug (e.g. print artefacts vs camera lighting).

        Args:
            foreground, background: image inputs. Either a numpy (H, W, 3)
                uint8/float array (single image) or a torch (B, C, H, W) /
                (C, H, W) float tensor. The foreground is the reference
                frame — if `background` / `mask` have a different spatial
                size they are resized to match the foreground (background
                bilinear, mask nearest). This lets layered compose chains
                work without the caller pre-sizing every input.
            mask: foreground-region mask. numpy (H, W) uint8/float or torch
                (B, 1, H, W) / (B, H, W) / (H, W). 1 (or 255) = foreground.
            seed_base, epoch, step: per-item RNG axes (see `__call__`).
            return_transform: if True, also return a `transform` dict
                describing the canvas stretch so callers with coordinate /
                angle GT stored outside the tensor can sync it.

        Returns:
            `(img, mask)` — or `(img, mask, transform)` if `return_transform`.
            Output type matches the `foreground` input type: numpy
            (H, W, C) uint8 in → numpy out; torch tensor in → torch out.

        Note:
            `compose` treats every foreground channel as image (photometric
            applies to all). Stack GT-as-channels through `__call__` with
            `n_image_channels` instead — that path is for GT that must skip
            photometric; `compose` is for the foreground/background/mask
            blend where the mask itself is the tracked GT.
        """
        fg, fg_was_numpy = _img_to_tensor(foreground)
        bg, _ = _img_to_tensor(background)
        m, _ = _mask_to_tensor(mask)

        if bg.shape[1] != fg.shape[1]:
            raise ValueError(
                f"foreground has {fg.shape[1]} channels but background has "
                f"{bg.shape[1]} — they must match for the blend")
        if not (fg.shape[0] == bg.shape[0] == m.shape[0]):
            raise ValueError(
                f"batch sizes differ: foreground {fg.shape[0]}, background "
                f"{bg.shape[0]}, mask {m.shape[0]}")

        dev = fg.device
        bg = bg.to(dev)
        m = m.to(dev)

        # Foreground is the reference frame. Resize background / mask to it
        # if they differ — makes layered compose chains (pass-1 output as
        # pass-2 foreground) work without the caller pre-sizing every input.
        fg_hw = fg.shape[-2:]
        if bg.shape[-2:] != fg_hw:
            bg = F.interpolate(bg, size=fg_hw, mode="bilinear",
                               align_corners=False)
        if m.shape[-2:] != fg_hw:
            m = F.interpolate(m, size=fg_hw, mode="nearest")

        pre_h, pre_w = fg_hw

        # Aug is preprocessing, never backpropped. See __call__ for why this
        # no_grad wrap matters (TL;DR: prevents VRAM blowup when caller's
        # inputs carry a live autograd graph).
        with torch.no_grad():
            # 1. geometric warp of (foreground, mask) — background stays static.
            fg_w, m_w = self._apply_geometric(fg, m, seed_base, epoch, step)
            # 2. blend foreground onto background through the warped mask.
            composite = fg_w * m_w + bg * (1.0 - m_w)
            composite = composite.clamp(0.0, 1.0)
            # 3. photometric on the composite (mask passes through untouched).
            composite, m_w = self._apply_photometric(
                composite, m_w, seed_base, epoch, step)
            # 4. conform to canvas.
            composite, m_w = self._resize_to_canvas(composite, m_w)

        transform = {
            "canvas_size": self.canvas_size,
            "scale_x": (self.canvas_size[1] / pre_w
                        if self.canvas_size is not None else 1.0),
            "scale_y": (self.canvas_size[0] / pre_h
                        if self.canvas_size is not None else 1.0),
        }

        if fg_was_numpy:
            img_out = _tensor_to_img_numpy(composite)
            mask_out = _tensor_to_mask_numpy(m_w)
        else:
            img_out = composite
            mask_out = m_w

        if return_transform:
            return img_out, mask_out, transform
        return img_out, mask_out
