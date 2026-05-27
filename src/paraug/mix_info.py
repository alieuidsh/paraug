"""Recover the per-item mixing factor λ and batch-partner index that paraug's
`cutmix` / `mixup` primitives sampled, so callers training classification
models can mix labels by the same λ.

paraug itself doesn't track labels — labels live in your training loop, and
paraug only sees the image tensor. To use CutMix or MixUp for classification:

    1. Configure paraug as usual (cutmix / mixup in the pipeline).
    2. Inside the train loop, BEFORE calling aug, recover λ and partner:

        from paraug import mix_info
        info = mix_info("cutmix", seed_base=step, epoch=epoch, step=step, B=B)
        # info["lam"]:    (B,) float — paraug's λ per item
        # info["perm"]:   (B,) long  — partner index per item
        # info["gate"]:   (B,) bool  — whether cutmix actually fired (per `p`)

    3. Call aug — paraug uses the same (seed_base, epoch, step) so it reaches
       the same λ and perm internally.

    4. Mix labels using the recovered values:

        labels_mix = info["lam"] * labels + (1 - info["lam"]) * labels[info["perm"]]
        # (or apply only where info["gate"] is True; same as paraug's gate)

The CutMix paper recomputes λ as the actual area ratio of the pasted region
(may differ from the sampled λ due to ceil() rounding). If you need that
exact ratio for a tight reproduction, call `mix_info("cutmix_actual_lam",
..., img_shape=(B, C, H, W))` — returns the post-rounding λ.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from .utils import cpu_generator, per_item_seed, sample_bool, sample_uniform


def _sample_lam_beta(alpha: float, g: torch.Generator) -> float:
    """Sample λ ~ Beta(α, α) using the same code path as cutmix/mixup
    (gamma trick from two uniforms, generator-aware)."""
    if abs(alpha - 1.0) < 1e-8:
        return float(torch.empty(1).uniform_(0, 1, generator=g).item())
    u1 = float(torch.empty(1).uniform_(0, 1, generator=g).item())
    u2 = float(torch.empty(1).uniform_(0, 1, generator=g).item())
    x1 = u1 ** (1.0 / alpha)
    x2 = u2 ** (1.0 / alpha)
    return x1 / (x1 + x2 + 1e-12)


def mix_info(
    primitive: str,
    seed_base: int,
    epoch: int,
    step: int,
    B: int,
    *,
    p: float = 1.0,
    alpha: float = 1.0,
    img_shape: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, torch.Tensor]:
    """Recover the per-item (λ, partner, gate) that paraug's mixing primitive
    sampled for this (seed_base, epoch, step). See module docstring.

    Args:
        primitive: "cutmix" or "mixup".
        seed_base, epoch, step: same triple passed to `aug.__call__`.
        B: batch size.
        p: per-item Bernoulli gate prob — must match the spec passed to paraug.
        alpha: Beta(α, α) shape — must match the spec passed to paraug.
            Defaults: cutmix=1.0, mixup=0.2 (paraug's own defaults).
        img_shape: (B, C, H, W). Only used when `primitive="cutmix_actual_lam"`
            to recompute the post-rounding area ratio.

    Returns:
        dict with keys ``lam`` (B,) float, ``perm`` (B,) long, ``gate`` (B,) bool.
        For ``primitive="cutmix_actual_lam"``, ``lam`` is the rounded area ratio
        of the pasted region (matches what cutmix actually applied to pixels).
    """
    if primitive not in ("cutmix", "mixup", "cutmix_actual_lam"):
        raise ValueError(f"primitive must be cutmix/mixup/cutmix_actual_lam; got {primitive!r}")
    base = "cutmix" if primitive.startswith("cutmix") else "mixup"

    gate = torch.zeros(B, dtype=torch.bool)
    lams = torch.ones(B, dtype=torch.float32)
    # For mixup paraug clips λ to [0.01, 0.99]; replicate here.
    clip_lam = (base == "mixup")

    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, base))
        if not sample_bool(p, g):
            continue
        gate[i] = True
        lam = _sample_lam_beta(alpha, g)
        if clip_lam:
            lam = max(0.01, min(0.99, lam))
        elif primitive == "cutmix_actual_lam":
            # Recompute as the actual pixel-area ratio after rounding.
            if img_shape is None:
                raise ValueError("cutmix_actual_lam requires img_shape=(B, C, H, W)")
            _, _, H, W = img_shape
            cut_frac = 1.0 - lam
            cut_h = max(1, int(round(H * math.sqrt(cut_frac))))
            cut_w = max(1, int(round(W * math.sqrt(cut_frac))))
            # The bounding box may clip to image; cutmix uses
            # cy = uniform(0, max(1, H - cut_h)), same here for fidelity.
            _ = sample_uniform(0, max(1, H - cut_h), g)  # advance RNG (cy)
            _ = sample_uniform(0, max(1, W - cut_w), g)  # advance RNG (cx)
            lam = 1.0 - (cut_h * cut_w) / float(H * W)
        # For plain cutmix we keep the sampled λ (caller mixes labels with it).
        lams[i] = lam

    g_perm = cpu_generator(per_item_seed(seed_base, epoch, step, 0, "cutmix_perm"))
    perm = torch.randperm(B, generator=g_perm)
    return {"lam": lams, "perm": perm, "gate": gate}
