"""Geometric primitives — both image and mask are warped together.

Each primitive has signature:
    fn(img, mask, spec, seed_base, epoch, step) -> (img_out, mask_out)

Image uses bilinear sampling; mask uses nearest. Padding is zero (background).
All primitives gate per-item: items with Bernoulli=False pass through bit-identical.
"""
import math
import torch
import torch.nn.functional as F

from .utils import per_item_seed, cpu_generator, sample_uniform, sample_bool


_GRID_CACHE = {}


def _coords(H: int, W: int, device, dtype):
    """Cached (yy, xx) pixel-coord meshgrid for sampling math."""
    key = (H, W, str(device), str(dtype))
    if key not in _GRID_CACHE:
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        _GRID_CACHE[key] = (yy, xx)
    return _GRID_CACHE[key]


def _apply_grid(img, mask, grid, gate_d):
    """Warp img + mask via grid; where gate is False, leave original.

    grid: (B, H, W, 2) in normalised [-1, 1] coords.
    gate_d: (B,) bool, on img.device.
    """
    B = img.shape[0]
    img_w = F.grid_sample(img, grid, mode="bilinear",
                          padding_mode="zeros", align_corners=True)
    img_out = torch.where(gate_d.view(B, 1, 1, 1).expand_as(img), img_w, img)
    if mask is not None:
        mask_w = F.grid_sample(mask, grid, mode="nearest",
                               padding_mode="zeros", align_corners=True)
        mask_out = torch.where(gate_d.view(B, 1, 1, 1).expand_as(mask), mask_w, mask)
    else:
        mask_out = None
    return img_out, mask_out


# ─── 1: tps ──────────────────────────────────────────────────────────
def _sample_tps_disp(B, H, W, p, max_disp, nc, seed_base, epoch, step, dtype):
    """Sample per-item TPS control points + upsample to full (H, W) displacement.

    Returns CPU tensors (caller should .to(device) as needed):
        gate    (B,) bool
        disp_y  (B, H, W)
        disp_x  (B, H, W)

    Factored out so any caller that wants to mirror tps()'s displacement
    sampling (e.g. a downstream pipeline that pairs tps with extra GT
    propagation) can pull the same disp field by using the same seed.
    """
    gate = torch.zeros(B, dtype=torch.bool)
    dy_ctrl = torch.zeros(B, 1, nc, nc, dtype=dtype)
    dx_ctrl = torch.zeros(B, 1, nc, nc, dtype=dtype)
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "tps"))
        if sample_bool(p, g):
            gate[i] = True
            dy_ctrl[i, 0] = torch.empty(nc, nc).uniform_(-max_disp, max_disp, generator=g)
            dx_ctrl[i, 0] = torch.empty(nc, nc).uniform_(-max_disp, max_disp, generator=g)
    disp_y = F.interpolate(dy_ctrl, size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    disp_x = F.interpolate(dx_ctrl, size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    return gate, disp_y, disp_x


def tps(img, mask, spec, seed_base, epoch, step):
    """Thin-plate-spline-ish warp via low-res control grid → bilinear upsample.

    spec = {"p": prob, "max_disp": pixels, "n_ctrl": int}
    """
    B, _, H, W = img.shape
    p = float(spec.get("p", 1.0))
    max_disp = float(spec.get("max_disp", 18.0))
    nc = int(spec.get("n_ctrl", 5))

    gate, disp_y, disp_x = _sample_tps_disp(
        B, H, W, p, max_disp, nc, seed_base, epoch, step, img.dtype)
    disp_y_d = disp_y.to(img.device); disp_x_d = disp_x.to(img.device)
    gate_d = gate.to(img.device)
    yy, xx = _coords(H, W, img.device, img.dtype)
    src_x = xx[None] - disp_x_d
    src_y = yy[None] - disp_y_d
    grid = torch.stack([2.0 * src_x / (W - 1) - 1.0,
                        2.0 * src_y / (H - 1) - 1.0], dim=-1)
    return _apply_grid(img, mask, grid, gate_d)


# ─── 2: affine ───────────────────────────────────────────────────────
def affine(img, mask, spec, seed_base, epoch, step):
    """Random rotation / scale / translation.

    spec = {"p": prob, "rot_deg": float, "scale_range": (lo, hi),
            "translate_frac": float (fraction of H/W)}
    """
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    rot_deg = float(spec.get("rot_deg", 30.0))
    scale_lo, scale_hi = spec.get("scale_range", (0.85, 1.15))
    trans_frac = float(spec.get("translate_frac", 0.05))

    gate = torch.zeros(B, dtype=torch.bool)
    # (B, 2, 3) affine matrix per item; identity for skipped items
    theta = torch.zeros(B, 2, 3, dtype=img.dtype)
    theta[:, 0, 0] = 1.0; theta[:, 1, 1] = 1.0
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "affine"))
        if sample_bool(p, g):
            gate[i] = True
            ang = math.radians(sample_uniform(-rot_deg, rot_deg, g))
            s = sample_uniform(scale_lo, scale_hi, g)
            tx = sample_uniform(-trans_frac, trans_frac, g)
            ty = sample_uniform(-trans_frac, trans_frac, g)
            ca, sa = math.cos(ang) / s, math.sin(ang) / s
            theta[i] = torch.tensor([[ca, sa, tx], [-sa, ca, ty]])
    theta_d = theta.to(img.device)
    gate_d = gate.to(img.device)
    grid = F.affine_grid(theta_d, [B, C, H, W], align_corners=True)
    return _apply_grid(img, mask, grid, gate_d)


# ─── 3: perspective ──────────────────────────────────────────────────
def perspective(img, mask, spec, seed_base, epoch, step):
    """4-point perspective via homography H built from corner jitters.

    spec = {"p": prob, "max_disp_frac": float (fraction of H/W)}
    Each corner of the image is jittered independently by up to ±max_disp_frac.
    """
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    max_disp = float(spec.get("max_disp_frac", 0.05))

    # Reference corners in normalised coords (-1, +1)
    src = torch.tensor([[-1.0, -1.0], [1.0, -1.0],
                        [1.0,  1.0], [-1.0, 1.0]], dtype=img.dtype)
    gate = torch.zeros(B, dtype=torch.bool)
    dst_all = src.unsqueeze(0).repeat(B, 1, 1)
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "perspective"))
        if sample_bool(p, g):
            gate[i] = True
            dst = src.clone()
            for c in range(4):
                dst[c, 0] += sample_uniform(-max_disp, max_disp, g) * 2  # *2 because normalised
                dst[c, 1] += sample_uniform(-max_disp, max_disp, g) * 2
            dst_all[i] = dst

    # Build homography per item (CPU, small linear solve), then bring grid to device
    yy, xx = _coords(H, W, "cpu", img.dtype)
    base_grid = torch.stack([2.0 * xx / (W - 1) - 1.0,
                             2.0 * yy / (H - 1) - 1.0], dim=-1)  # (H, W, 2) in [-1, 1]
    grids = torch.empty(B, H, W, 2, dtype=img.dtype)
    for i in range(B):
        Hmat = _solve_homography(src, dst_all[i])
        # Apply homography to base_grid pixels (back-warp)
        bg = base_grid.reshape(-1, 2)  # (HW, 2)
        ones = torch.ones(bg.shape[0], 1, dtype=img.dtype)
        bgh = torch.cat([bg, ones], dim=1)  # (HW, 3)
        warped = bgh @ Hmat.T
        warped = warped[:, :2] / warped[:, 2:].clamp(min=1e-8)
        grids[i] = warped.reshape(H, W, 2)
    grid = grids.to(img.device)
    gate_d = gate.to(img.device)
    return _apply_grid(img, mask, grid, gate_d)


def _solve_homography(src, dst):
    """Compute 3×3 homography mapping src corners to dst corners. CPU, fp32.
    src, dst: (4, 2) in normalised coords."""
    A = torch.zeros(8, 8, dtype=src.dtype)
    b = torch.zeros(8, dtype=src.dtype)
    for k in range(4):
        x, y = src[k]
        u, v = dst[k]
        A[2*k]     = torch.tensor([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A[2*k + 1] = torch.tensor([0, 0, 0, x, y, 1, -v*x, -v*y])
        b[2*k]     = u
        b[2*k + 1] = v
    h = torch.linalg.solve(A, b)
    H = torch.tensor([[h[0], h[1], h[2]],
                      [h[3], h[4], h[5]],
                      [h[6], h[7], 1.0]], dtype=src.dtype)
    return H


# ─── 4: random_crop_pad ──────────────────────────────────────────────
def random_crop_pad(img, mask, spec, seed_base, epoch, step):
    """Random crop, then pad back to original size with zero (or reflect).

    spec = {"p": prob, "min_scale": float (lower bound on crop area frac)}
    """
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    min_scale = float(spec.get("min_scale", 0.7))

    # We build the sampling grid: identity scaled + translated within bounds.
    gate = torch.zeros(B, dtype=torch.bool)
    theta = torch.zeros(B, 2, 3, dtype=img.dtype)
    theta[:, 0, 0] = 1.0; theta[:, 1, 1] = 1.0
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "random_crop_pad"))
        if sample_bool(p, g):
            gate[i] = True
            s = sample_uniform(min_scale, 1.0, g)
            # crop area = s × original; translation within (1-s) so crop stays inside
            tx = sample_uniform(-(1 - s), (1 - s), g)
            ty = sample_uniform(-(1 - s), (1 - s), g)
            theta[i] = torch.tensor([[s, 0, tx], [0, s, ty]])
    theta_d = theta.to(img.device); gate_d = gate.to(img.device)
    grid = F.affine_grid(theta_d, [B, C, H, W], align_corners=True)
    return _apply_grid(img, mask, grid, gate_d)


# ─── 5: elastic_transform ────────────────────────────────────────────
def elastic_transform(img, mask, spec, seed_base, epoch, step):
    """Elastic displacement-field warp.

    spec = {"p": prob, "alpha": pixels (max disp)}

    Implementation note (vs albumentations' ElasticTransform):
    The displacement field is bilinear-upsampled from a low-resolution random
    grid of shape (H/8, W/8). This differs from albumentations' dense
    Gaussian-smoothed noise; ours produces smoother but more grid-aligned
    distortion. Empirically indistinguishable for sim-to-real augmentation.
    Smoothness is controlled by the fixed 1/8 downsample factor (no `sigma`
    knob — by-design).
    """
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    alpha = float(spec.get("alpha", 8.0))

    gate = torch.zeros(B, dtype=torch.bool)
    # Per-item random field at full resolution, smoothed via separable Gaussian
    disp_y = torch.zeros(B, H, W, dtype=img.dtype)
    disp_x = torch.zeros(B, H, W, dtype=img.dtype)
    # Build a low-res random field (H/8, W/8) and bicubic-upsample for speed +
    # smoothness (equivalent to smoothing with σ proportional to scale factor).
    hl = max(8, int(H / 8)); wl = max(8, int(W / 8))
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "elastic_transform"))
        if sample_bool(p, g):
            gate[i] = True
            dy_low = torch.empty(1, 1, hl, wl).uniform_(-1.0, 1.0, generator=g)
            dx_low = torch.empty(1, 1, hl, wl).uniform_(-1.0, 1.0, generator=g)
            # Upsample bilinear, scale by alpha
            disp_y[i] = F.interpolate(dy_low, size=(H, W), mode="bilinear", align_corners=False)[0, 0] * alpha
            disp_x[i] = F.interpolate(dx_low, size=(H, W), mode="bilinear", align_corners=False)[0, 0] * alpha
            # Optional: gaussian smooth — keep simple with bilinear upsample only
    disp_y_d = disp_y.to(img.device); disp_x_d = disp_x.to(img.device)
    gate_d = gate.to(img.device)
    yy, xx = _coords(H, W, img.device, img.dtype)
    src_x = xx[None] - disp_x_d
    src_y = yy[None] - disp_y_d
    grid = torch.stack([2.0 * src_x / (W - 1) - 1.0,
                        2.0 * src_y / (H - 1) - 1.0], dim=-1)
    return _apply_grid(img, mask, grid, gate_d)


# ─── 6: optical_distortion ───────────────────────────────────────────
def optical_distortion(img, mask, spec, seed_base, epoch, step):
    """Barrel (k>0) / pincushion (k<0) lens distortion.

    spec = {"p": prob, "k": float coefficient (max abs value)}
    Maps (x, y) → (x', y') = (x·(1 + k·r²), y·(1 + k·r²)), r = normalised radius.
    Per-item k sampled uniformly from [-k_max, k_max].

    Implementation note:
    This uses a first-order **forward** approximation `scale = 1 + k·r²` for
    the back-warp lookup (computing src = nx · scale using output r). For
    moderate k ∈ [-0.5, 0.5] the visual distortion is geometrically correct.
    For larger |k| the strictly-correct inverse requires Newton iteration on
    a cubic; not implemented since the small-k regime covers typical
    photographic distortion.
    """
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    kmax = float(spec.get("k", 0.3))

    gate = torch.zeros(B, dtype=torch.bool)
    ks = torch.zeros(B, dtype=img.dtype)
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "optical_distortion"))
        if sample_bool(p, g):
            gate[i] = True
            ks[i] = sample_uniform(-kmax, kmax, g)
    ks_d = ks.to(img.device); gate_d = gate.to(img.device)

    yy, xx = _coords(H, W, img.device, img.dtype)
    nx = 2.0 * xx / (W - 1) - 1.0
    ny = 2.0 * yy / (H - 1) - 1.0
    r2 = nx ** 2 + ny ** 2  # (H, W)
    # Per-item radial scale: 1 + k_i * r²
    scale = 1.0 + ks_d.view(B, 1, 1) * r2[None]
    src_nx = nx[None] * scale  # back-warp coords in normalised space
    src_ny = ny[None] * scale
    grid = torch.stack([src_nx, src_ny], dim=-1)
    return _apply_grid(img, mask, grid, gate_d)


# ─── 7: random_shadow (vectorized) ────────────────────────────────────
def random_shadow(img, mask, spec, seed_base, epoch, step):
    """Random triangular shadow region — multiplies image by a soft falloff
    factor inside a randomly placed triangle. Mask is unchanged.

    spec = {"p": prob, "strength": float in [0,1], "softness_px": float}

    Vectorized version (Phase 4): the previous per-item Python loop over the
    avg_pool blur (3 passes × kernel ≈ 61 per item × B items) was the worst
    wall-clock primitive at ~4.2 s/iter on B=8 850×1100. Batching the box blur
    over the B dimension lets the GPU run all items in parallel; the per-item
    triangle is built with broadcasted barycentric math instead of a Python
    loop. RNG sampling stays per-item to preserve bit-exact determinism.
    """
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    strength = float(spec.get("strength", 0.4))
    softness = float(spec.get("softness_px", 30.0))

    # Per-item RNG sampling (small loop, cheap)
    gate = torch.zeros(B, dtype=torch.bool)
    pts_all = torch.zeros(B, 3, 2, dtype=img.dtype)
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "random_shadow"))
        if sample_bool(p, g):
            gate[i] = True
            for k in range(3):
                pts_all[i, k, 0] = sample_uniform(0, W - 1, g)
                pts_all[i, k, 1] = sample_uniform(0, H - 1, g)
    pts_d = pts_all.to(img.device)
    gate_d = gate.to(img.device)

    # Vectorized point-in-triangle barycentric, broadcast (B, H, W)
    yy, xx = _coords(H, W, img.device, img.dtype)
    p0x = pts_d[:, 0, 0].view(B, 1, 1); p0y = pts_d[:, 0, 1].view(B, 1, 1)
    p1x = pts_d[:, 1, 0].view(B, 1, 1); p1y = pts_d[:, 1, 1].view(B, 1, 1)
    p2x = pts_d[:, 2, 0].view(B, 1, 1); p2y = pts_d[:, 2, 1].view(B, 1, 1)
    denom = (p1y - p2y) * (p0x - p2x) + (p2x - p1x) * (p0y - p2y)  # (B, 1, 1)
    # Avoid div by zero for degenerate (non-gated) items — their result gets
    # masked out by the slot-gate further below.
    denom_safe = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)
    xx_b = xx.unsqueeze(0); yy_b = yy.unsqueeze(0)
    a = ((p1y - p2y) * (xx_b - p2x) + (p2x - p1x) * (yy_b - p2y)) / denom_safe
    b = ((p2y - p0y) * (xx_b - p2x) + (p0x - p2x) * (yy_b - p2y)) / denom_safe
    c = 1.0 - a - b
    inside = (a >= 0) & (b >= 0) & (c >= 0) & (denom.abs() >= 1e-8)
    mask_in = inside.to(img.dtype)  # (B, H, W)

    # 3-pass iterated box blur on the whole batch in one shot (GPU friendly).
    kbox = max(3, int(2 * softness) | 1)
    padbox = kbox // 2
    soft = mask_in.unsqueeze(1)  # (B, 1, H, W)
    for _ in range(3):
        soft = F.avg_pool2d(soft, kernel_size=kbox, stride=1, padding=padbox)
    shadow_factor = 1.0 - strength * soft.clamp(0, 1)  # (B, 1, H, W)

    img_dim = img * shadow_factor
    img_out = torch.where(gate_d.view(B, 1, 1, 1).expand_as(img), img_dim, img)
    return img_out, mask


GEOMETRIC_PRIMITIVES = {
    "tps": tps,
    "affine": affine,
    "perspective": perspective,
    "random_crop_pad": random_crop_pad,
    "elastic_transform": elastic_transform,
    "optical_distortion": optical_distortion,
    "random_shadow": random_shadow,
}
