"""Layout helpers — geometric placement of a foreground inside a larger
canvas before composing.

Most aug pipelines model "what happens to the image" — colour, blur, warp.
`paraug.layout` models "where the content sits inside the frame": is the
content centred, off to the left, padded with a thick top margin? This
turns out to matter when the segmentation target is a **sub-region** of
the foreground (e.g. an ECG content rectangle inside a larger paper sheet)
because without random placement the model never learns that the wider
paper frame is a *distractor* it should ignore.

`place_into_canvas` is the main helper. Use it as a pre-step before
`AugPipeline.compose` to randomise where the foreground sits inside a
paper-sized canvas:

    from paraug import place_into_canvas, AugPipeline

    # ECG content + ECG-region mask
    ecg_padded, ecg_padded_mask = place_into_canvas(
        ecg_content, ecg_inner_mask,
        canvas_size=(800, 1000),
        fill=paper_tone_rgb,             # white-paper colour
        margin_frac_range=(0.05, 0.30),  # 5-30% white margin per side
        seed_base=epoch_step_seed,
    )
    # Now compose: the "foreground" is a paper-sized sheet with ECG inside
    img, mask = aug.compose(ecg_padded, scene_bg, paper_outline_mask)

All sampling uses paraug's CPU-side per-item RNG (`cpu_generator`,
`per_item_seed`) so the placement is deterministic and CPU/CUDA-identical
under the same seed.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .utils import cpu_generator, per_item_seed, sample_uniform


def _as_tensor_img(x: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, bool]:
    """Like pipeline._img_to_tensor but local — avoids cross-module import
    pulling photometric primitives just to do shape conversion."""
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(x))
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            t = t.float()
        if t.dim() == 2:
            t = t[None, None]
        elif t.dim() == 3:
            t = t.permute(2, 0, 1)[None].contiguous()
        else:
            raise ValueError(
                f"numpy image must be (H,W) or (H,W,C); got shape {x.shape}")
        return t, True
    t = x
    if t.dim() == 3:
        t = t[None]
    elif t.dim() != 4:
        raise ValueError(
            f"torch image must be (C,H,W) or (B,C,H,W); got shape {tuple(t.shape)}")
    return t.float(), False


def _as_tensor_mask(x: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, bool]:
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(x))
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            t = t.float()
        if t.dim() == 2:
            t = t[None, None]
        elif t.dim() == 3:
            t = t.permute(2, 0, 1)[None].contiguous()
        else:
            raise ValueError(
                f"numpy mask must be (H,W) or (H,W,1); got shape {x.shape}")
        return t, True
    t = x
    if t.dim() == 2:
        t = t[None, None]
    elif t.dim() == 3:
        t = t[:, None]
    elif t.dim() != 4:
        raise ValueError(
            f"torch mask must be (H,W)/(B,H,W)/(B,1,H,W); got {tuple(t.shape)}")
    return t.float(), False


def _normalise_fill(fill, channels: int, dtype, device) -> torch.Tensor:
    """Coerce `fill` into a (channels,) float tensor in [0, 1].
    Accepts a Python scalar, a sequence of length=channels, or a tensor."""
    if fill is None:
        fill = 1.0  # default = white
    if isinstance(fill, (int, float)):
        t = torch.full((channels,), float(fill), dtype=dtype, device=device)
    elif isinstance(fill, (tuple, list)):
        if len(fill) != channels:
            raise ValueError(
                f"fill sequence length {len(fill)} != channels {channels}")
        t = torch.tensor(list(fill), dtype=dtype, device=device)
    elif isinstance(fill, np.ndarray):
        t = torch.from_numpy(np.asarray(fill)).to(dtype=dtype, device=device)
        if t.numel() != channels:
            raise ValueError(
                f"fill array length {t.numel()} != channels {channels}")
    elif torch.is_tensor(fill):
        t = fill.to(dtype=dtype, device=device)
        if t.numel() != channels:
            raise ValueError(
                f"fill tensor length {t.numel()} != channels {channels}")
    else:
        raise TypeError(f"unsupported fill type {type(fill).__name__}")
    if t.dtype == torch.uint8 or (t.max() > 1.5).item():
        t = t.float() / 255.0
    return t.clamp(0.0, 1.0)


def place_into_canvas(
    foreground: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
    canvas_size: Tuple[int, int],
    fill=1.0,
    position: str = "random",
    margin_frac_range: Tuple[float, float] = (0.05, 0.25),
    seed_base: int = 0,
    epoch: int = 0,
    step: int = 0,
):
    """Place a foreground (and its mask) at some position inside a larger
    canvas, surrounded by a constant-colour fill (e.g. paper tone).

    Critically the *mask* is preserved untouched and zero-padded outside the
    foreground — so downstream training sees "ECG content at position
    (cx, cy) within a larger paper canvas, with the loss target restricted
    to the actual ECG region". Without this aug, every training sample has
    the same content-fills-the-frame layout and the model never learns that
    a larger surrounding rectangle (the paper itself, then the desk under
    the paper) is a distractor it must ignore.

    The output has the foreground type (numpy in → numpy out, tensor in →
    tensor out). Returns `(canvas_img, canvas_mask)` with spatial size
    exactly `canvas_size = (H, W)`.

    Args:
        foreground: image to place. numpy (H, W, C) uint8/float, torch
            (B, C, H, W) / (C, H, W).
        mask: aligned mask. Same convention as `AugPipeline.compose`.
        canvas_size: (H, W) of the output canvas. Must be >= foreground
            size after the chosen margin is applied; the helper down-scales
            the foreground if it does not fit.
        fill: background fill colour for the canvas. Either a scalar
            (broadcast to all channels — e.g. `1.0` for white paper),
            a length-C sequence/tensor of per-channel values in [0, 1] or
            [0, 255] uint8.
        position: how to choose the foreground anchor within the canvas:
            * `"random"` — uniformly sample x and y offsets that keep the
              foreground inside the canvas after honouring `margin_frac_range`
            * `"center"` — centre the foreground (deterministic, ignores
              `margin_frac_range`)
        margin_frac_range: (lo, hi). For each axis independently, the
            minimum margin (as a fraction of canvas size on that axis) the
            foreground anchor must leave on each side. e.g. `(0.05, 0.30)`
            requires at least 5% margin on each side and lets up to 30% be
            left over. Honoured only when `position="random"`.
        seed_base, epoch, step: paraug-standard per-item RNG axes. Each
            batch item uses its own deterministic offset.

    Returns:
        `(canvas_img, canvas_mask)` matching the foreground input type.
    """
    if position not in ("random", "center"):
        raise ValueError(f"position must be 'random' or 'center'; got {position!r}")
    if not (isinstance(canvas_size, (tuple, list)) and len(canvas_size) == 2
            and int(canvas_size[0]) >= 1 and int(canvas_size[1]) >= 1):
        raise ValueError(f"canvas_size must be (H, W) with H, W >= 1; got {canvas_size!r}")
    canvas_h, canvas_w = int(canvas_size[0]), int(canvas_size[1])

    fg, fg_was_numpy = _as_tensor_img(foreground)
    m, _ = _as_tensor_mask(mask)
    if fg.shape[-2:] != m.shape[-2:]:
        raise ValueError(
            f"foreground {tuple(fg.shape[-2:])} and mask {tuple(m.shape[-2:])} "
            f"must have the same spatial size")
    if fg.shape[0] != m.shape[0]:
        raise ValueError(
            f"batch sizes differ: foreground {fg.shape[0]}, mask {m.shape[0]}")

    B, C, in_h, in_w = fg.shape
    dev = fg.device
    fg = fg.to(dev)
    m = m.to(dev)
    fill_t = _normalise_fill(fill, C, fg.dtype, dev).view(1, C, 1, 1)

    # Build the canvas (B, C, H, W) filled with the constant colour, and a
    # zero mask canvas. Then for each item sample its placement and write
    # the (possibly resized) foreground in.
    canvas_img = fill_t.expand(B, C, canvas_h, canvas_w).contiguous().clone()
    canvas_mask = torch.zeros(B, 1, canvas_h, canvas_w, dtype=m.dtype, device=dev)

    margin_lo, margin_hi = float(margin_frac_range[0]), float(margin_frac_range[1])
    if not (0.0 <= margin_lo <= margin_hi < 0.5):
        raise ValueError(
            f"margin_frac_range must be (lo, hi) with 0 <= lo <= hi < 0.5; "
            f"got {margin_frac_range!r}")

    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "place_into_canvas"))
        # Per-item margin fraction (uniform in [lo, hi] independently per axis)
        if position == "random":
            margin_h_frac = sample_uniform(margin_lo, margin_hi, g)
            margin_w_frac = sample_uniform(margin_lo, margin_hi, g)
        else:
            margin_h_frac = margin_w_frac = 0.0

        # Available size for foreground after honouring margins.
        avail_h = max(1, int(canvas_h * (1.0 - 2.0 * margin_h_frac)))
        avail_w = max(1, int(canvas_w * (1.0 - 2.0 * margin_w_frac)))
        # Scale foreground down (preserving its own aspect ratio) so it fits.
        sx = avail_w / in_w
        sy = avail_h / in_h
        s = min(sx, sy, 1.0)  # never upscale — only down-fit
        out_h = max(1, int(round(in_h * s)))
        out_w = max(1, int(round(in_w * s)))

        if (out_h, out_w) != (in_h, in_w):
            fg_i = F.interpolate(fg[i:i + 1], size=(out_h, out_w),
                                  mode="bilinear", align_corners=False)
            m_i = F.interpolate(m[i:i + 1], size=(out_h, out_w),
                                 mode="nearest")
        else:
            fg_i = fg[i:i + 1]
            m_i = m[i:i + 1]

        # Anchor position
        free_h = canvas_h - out_h
        free_w = canvas_w - out_w
        if position == "random":
            top = int(sample_uniform(0, free_h, g)) if free_h > 0 else 0
            left = int(sample_uniform(0, free_w, g)) if free_w > 0 else 0
        else:
            top = free_h // 2
            left = free_w // 2

        canvas_img[i:i + 1, :, top:top + out_h, left:left + out_w] = fg_i
        canvas_mask[i:i + 1, :, top:top + out_h, left:left + out_w] = m_i

    if fg_was_numpy:
        img_np = canvas_img[0].clamp(0, 1).mul(255).round().to(torch.uint8)
        img_np = img_np.permute(1, 2, 0).contiguous().cpu().numpy()
        mask_np = canvas_mask[0, 0].clamp(0, 1).mul(255).round().to(torch.uint8)
        mask_np = mask_np.contiguous().cpu().numpy()
        return img_np, mask_np
    return canvas_img, canvas_mask
