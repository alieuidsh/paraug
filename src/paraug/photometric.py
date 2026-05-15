"""Photometric primitives — operate on RGB image tensor; mask passes through unchanged.

Each primitive has signature:
    fn(img, mask, spec, seed_base, epoch, step) -> (img_out, mask_out)

`img` is (B, 3, H, W) float in [0, 1], any device. `mask` is (B, 1, H, W) or None.
All randomness is CPU-side, samples then moved to img.device, ensuring CPU/CUDA
bit-exact within fp32 ulp.
"""
import math
import torch

from .utils import per_item_seed, cpu_generator, sample_uniform, sample_bool


# ─── 13: gamma ────────────────────────────────────────────────────────
def gamma(img, mask, spec, seed_base, epoch, step):
    """Per-item gamma correction: out = img.clamp(0,1) ^ γ, γ ∈ gamma_range.

    spec = {"p": float in [0,1], "gamma_range": (float, float)}

    Bit-exact: elementwise op, tolerance class 1e-6 (torch.pow is elementwise on
    CPU and CUDA backed by the same IEEE-754 fp32 semantics).

    Items that fail the per-item Bernoulli test pass through bit-identical
    (via torch.where on a per-item gate, avoiding pow(1.0) round-trip).
    """
    B = img.shape[0]
    p = float(spec.get("p", 1.0))
    gmin, gmax = spec.get("gamma_range", (0.6, 1.6))

    # Per-item: Bernoulli gate + gamma value
    gate = torch.zeros(B, dtype=torch.bool)
    gammas = torch.ones(B, dtype=img.dtype)
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "gamma"))
        if sample_bool(p, g):
            gate[i] = True
            gammas[i] = sample_uniform(gmin, gmax, g)
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    gammas_d = gammas.to(img.device).view(B, 1, 1, 1)
    # Compute powered version; mask gate selects per-item
    powered = img.clamp(0.0, 1.0).pow(gammas_d)
    out = torch.where(gate_d, powered, img)
    return out, mask


# ─── batch 1: simple elementwise ──────────────────────────────────────
def gaussian_noise(img, mask, spec, seed_base, epoch, step):
    """Per-item additive Gaussian noise with σ sampled from [lo, hi].

    spec = {"p": prob, "sigma_range": (lo, hi)}
    """
    B = img.shape[0]
    p = float(spec.get("p", 1.0))
    smin, smax = spec.get("sigma_range", (0.005, 0.03))
    gate = torch.zeros(B, dtype=torch.bool)
    noise = torch.zeros_like(img.cpu()) if img.is_cuda else torch.zeros_like(img)
    sigmas = torch.zeros(B, dtype=img.dtype)
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "gaussian_noise"))
        if sample_bool(p, g):
            gate[i] = True
            sigmas[i] = sample_uniform(smin, smax, g)
            noise[i] = torch.empty_like(img[0].cpu() if img.is_cuda else img[0]).normal_(0.0, 1.0, generator=g) * sigmas[i]
    noise_d = noise.to(img.device)
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    out = torch.where(gate_d.expand_as(img), (img + noise_d).clamp(0, 1), img)
    return out, mask


def salt_pepper_noise(img, mask, spec, seed_base, epoch, step):
    """Per-pixel salt/pepper: with prob `density`, replace pixel with 0 or 1.

    spec = {"p": prob, "density": fraction of pixels affected, "salt_vs_pepper": float [0,1]}
    """
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    density = float(spec.get("density", 0.01))
    salt_frac = float(spec.get("salt_vs_pepper", 0.5))
    gate = torch.zeros(B, dtype=torch.bool)
    # Per-pixel uniform mask, per-item; we sample (B, 1, H, W) on CPU
    pix_rand = torch.zeros(B, 1, H, W, dtype=img.dtype)
    salt_choice = torch.zeros(B, 1, H, W, dtype=img.dtype)
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "salt_pepper_noise"))
        if sample_bool(p, g):
            gate[i] = True
            pix_rand[i, 0] = torch.empty(H, W).uniform_(0.0, 1.0, generator=g)
            salt_choice[i, 0] = torch.empty(H, W).uniform_(0.0, 1.0, generator=g)
    pix_d = pix_rand.to(img.device)
    salt_d = salt_choice.to(img.device)
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    affected = pix_d < density          # bool mask of pixels to change
    is_salt = salt_d < salt_frac        # salt vs pepper choice
    val = torch.where(is_salt, torch.ones_like(img), torch.zeros_like(img))
    new_img = torch.where(affected.expand_as(img), val, img)
    out = torch.where(gate_d.expand_as(img), new_img, img)
    return out, mask


def color_jitter(img, mask, spec, seed_base, epoch, step):
    """Per-item brightness · contrast · saturation jitter.

    spec = {"p": prob, "brightness": float (±frac), "contrast": float, "saturation": float}
    Order: brightness → contrast → saturation (matches torchvision convention).
    """
    B = img.shape[0]
    p = float(spec.get("p", 1.0))
    br = float(spec.get("brightness", 0.2))
    ct = float(spec.get("contrast",   0.2))
    sa = float(spec.get("saturation", 0.2))
    gate = torch.zeros(B, dtype=torch.bool)
    bvals = torch.ones(B, dtype=img.dtype)
    cvals = torch.ones(B, dtype=img.dtype)
    svals = torch.ones(B, dtype=img.dtype)
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "color_jitter"))
        if sample_bool(p, g):
            gate[i] = True
            bvals[i] = sample_uniform(1.0 - br, 1.0 + br, g)
            cvals[i] = sample_uniform(1.0 - ct, 1.0 + ct, g)
            svals[i] = sample_uniform(1.0 - sa, 1.0 + sa, g)
    b_d = bvals.to(img.device).view(B, 1, 1, 1)
    c_d = cvals.to(img.device).view(B, 1, 1, 1)
    s_d = svals.to(img.device).view(B, 1, 1, 1)
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    # brightness: multiply
    x = img * b_d
    # contrast: about the mean
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * c_d + mean
    # saturation: blend with grayscale
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = gray + s_d * (x - gray)
    x = x.clamp(0, 1)
    out = torch.where(gate_d.expand_as(img), x, img)
    return out, mask


def random_grayscale(img, mask, spec, seed_base, epoch, step):
    """Per-item replace RGB with grayscale (broadcast to 3 channels).

    spec = {"p": prob} — applied at given probability.
    """
    B, C, _, _ = img.shape
    p = float(spec.get("p", 0.3))
    gate = torch.zeros(B, dtype=torch.bool)
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "random_grayscale"))
        if sample_bool(p, g):
            gate[i] = True
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    gray = (0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]).expand_as(img)
    out = torch.where(gate_d.expand_as(img), gray, img)
    return out, mask


def cutout(img, mask, spec, seed_base, epoch, step):
    """Per-item rectangular cutout(s) replaced with constant value.

    spec = {"p": prob, "n_boxes_range": (lo, hi), "size_frac_range": (lo, hi),
            "value": float (fill value)}
    """
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    nlo, nhi = spec.get("n_boxes_range", (1, 3))
    slo, shi = spec.get("size_frac_range", (0.05, 0.2))
    val = float(spec.get("value", 0.5))
    gate = torch.zeros(B, dtype=torch.bool)
    fill_mask = torch.zeros(B, 1, H, W, dtype=img.dtype)
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "cutout"))
        if sample_bool(p, g):
            gate[i] = True
            n = int(torch.empty(1).uniform_(nlo, nhi + 1, generator=g).item())
            for _ in range(n):
                sh = int(sample_uniform(slo, shi, g) * H)
                sw = int(sample_uniform(slo, shi, g) * W)
                cy = int(sample_uniform(0, H - sh, g))
                cx = int(sample_uniform(0, W - sw, g))
                fill_mask[i, 0, cy:cy + sh, cx:cx + sw] = 1.0
    fill_d = fill_mask.to(img.device)
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    filled = torch.where(fill_d.expand_as(img) > 0.5, torch.full_like(img, val), img)
    out = torch.where(gate_d.expand_as(img), filled, img)
    return out, mask


# ─── batch 2: pattern overlay (elementwise after pattern build) ───────
def _xy_coords(H, W, dtype, device="cpu"):
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=dtype, device=device),
        torch.arange(W, dtype=dtype, device=device),
        indexing="ij")
    return yy, xx


def lighting(img, mask, spec, seed_base, epoch, step):
    """2D Gaussian lighting gradient: bright/dim spot multiplies the image.

    spec = {"p": prob, "strength": float (peak +/- multiplier),
            "sigma_frac": float (Gaussian σ as fraction of min(H, W))}
    """
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    strength = float(spec.get("strength", 0.2))
    sigma_frac = float(spec.get("sigma_frac", 0.4))
    sigma = sigma_frac * min(H, W)
    gate = torch.zeros(B, dtype=torch.bool)
    factor = torch.ones(B, 1, H, W, dtype=img.dtype)
    yy, xx = _xy_coords(H, W, img.dtype)
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "lighting"))
        if sample_bool(p, g):
            gate[i] = True
            cy = sample_uniform(0, H - 1, g); cx = sample_uniform(0, W - 1, g)
            amp = sample_uniform(-strength, strength, g)
            r2 = (yy - cy) ** 2 + (xx - cx) ** 2
            gaus = torch.exp(-r2 / (2 * sigma * sigma))
            factor[i, 0] = 1.0 + amp * gaus
    factor_d = factor.to(img.device)
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    new_img = (img * factor_d).clamp(0, 1)
    out = torch.where(gate_d.expand_as(img), new_img, img)
    return out, mask


def vignette(img, mask, spec, seed_base, epoch, step):
    """Radial darkening from center. factor = 1 - strength · (r/r_max)^power.

    spec = {"p": prob, "strength_range": (lo, hi), "power_range": (lo, hi)}
    """
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    smin, smax = spec.get("strength_range", (0.2, 0.5))
    pmin, pmax = spec.get("power_range", (2.0, 4.0))
    gate = torch.zeros(B, dtype=torch.bool)
    factor = torch.ones(B, 1, H, W, dtype=img.dtype)
    yy, xx = _xy_coords(H, W, img.dtype)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r_max = math.sqrt(cy ** 2 + cx ** 2)
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / r_max
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "vignette"))
        if sample_bool(p, g):
            gate[i] = True
            s = sample_uniform(smin, smax, g)
            pow_ = sample_uniform(pmin, pmax, g)
            factor[i, 0] = 1.0 - s * (r ** pow_)
    factor_d = factor.to(img.device).clamp(0, 1)
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    new_img = (img * factor_d).clamp(0, 1)
    out = torch.where(gate_d.expand_as(img), new_img, img)
    return out, mask


def salt_patches(img, mask, spec, seed_base, epoch, step):
    """Small bright round spots (water/rain droplets). Vectorized over B."""
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    nlo, nhi = spec.get("n_range", (3, 10))
    rmin, rmax = spec.get("radius_range", (2, 6))
    imin, imax = spec.get("intensity_range", (0.5, 1.0))
    n_max = int(nhi)

    # Per-item RNG → pre-sample all slot params
    gate = torch.zeros(B, dtype=torch.bool)
    n_per_item = torch.zeros(B, dtype=torch.long)
    cy_t = torch.zeros(B, n_max, dtype=img.dtype)
    cx_t = torch.zeros(B, n_max, dtype=img.dtype)
    r_t  = torch.ones(B, n_max, dtype=img.dtype)  # default 1 to avoid div by zero
    int_t = torch.zeros(B, n_max, dtype=img.dtype)
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "salt_patches"))
        if sample_bool(p, g):
            gate[i] = True
            n = int(torch.empty(1).uniform_(nlo, nhi + 1, generator=g).item())
            n_per_item[i] = n
            for s in range(n):
                cy_t[i, s] = sample_uniform(0, H - 1, g)
                cx_t[i, s] = sample_uniform(0, W - 1, g)
                r_t[i, s] = sample_uniform(rmin, rmax, g)
                int_t[i, s] = sample_uniform(imin, imax, g)
    cy_d = cy_t.to(img.device); cx_d = cx_t.to(img.device)
    r_d = r_t.to(img.device); int_d = int_t.to(img.device)
    nper_d = n_per_item.to(img.device)
    gate_d = gate.to(img.device)

    yy, xx = _xy_coords(H, W, img.dtype, img.device)
    add = torch.zeros(B, H, W, dtype=img.dtype, device=img.device)
    for s in range(n_max):
        valid = (s < nper_d).view(B, 1, 1)  # (B, 1, 1) bool — items where this slot is active
        cy_s = cy_d[:, s].view(B, 1, 1); cx_s = cx_d[:, s].view(B, 1, 1)
        r_s  = r_d[:, s].view(B, 1, 1);  i_s  = int_d[:, s].view(B, 1, 1)
        r2 = (yy.unsqueeze(0) - cy_s) ** 2 + (xx.unsqueeze(0) - cx_s) ** 2
        blob = torch.exp(-r2 / (2 * r_s * r_s)) * i_s
        blob = torch.where(valid, blob, torch.zeros_like(blob))
        add = torch.maximum(add, blob)
    add_d = add.unsqueeze(1)  # (B, 1, H, W)
    gate_4d = gate_d.view(B, 1, 1, 1)
    new_img = (img + add_d.expand_as(img)).clamp(0, 1)
    out = torch.where(gate_4d.expand_as(img), new_img, img)
    return out, mask


def watermark(img, mask, spec, seed_base, epoch, step):
    """Semi-transparent rectangular blob overlay. Vectorized over B."""
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    slo, shi = spec.get("size_frac_range", (0.1, 0.3))
    alo, ahi = spec.get("alpha_range", (0.2, 0.5))
    vlo, vhi = spec.get("value_range", (0.3, 0.8))

    # Per-item RNG: pre-sample rectangle params
    gate = torch.zeros(B, dtype=torch.bool)
    sh_t = torch.zeros(B, dtype=torch.long)
    sw_t = torch.zeros(B, dtype=torch.long)
    cy_t = torch.zeros(B, dtype=torch.long)
    cx_t = torch.zeros(B, dtype=torch.long)
    alphas = torch.zeros(B, dtype=img.dtype)
    vals = torch.zeros(B, dtype=img.dtype)
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "watermark"))
        if sample_bool(p, g):
            gate[i] = True
            sh_t[i] = int(sample_uniform(slo, shi, g) * H)
            sw_t[i] = int(sample_uniform(slo, shi, g) * W)
            cy_t[i] = int(sample_uniform(0, max(1, H - sh_t[i].item()), g))
            cx_t[i] = int(sample_uniform(0, max(1, W - sw_t[i].item()), g))
            alphas[i] = sample_uniform(alo, ahi, g)
            vals[i] = sample_uniform(vlo, vhi, g)

    # Vectorized mask build via broadcast comparison (B, H, W)
    yy, xx = _xy_coords(H, W, img.dtype, img.device)
    cy_d = cy_t.to(img.device).view(B, 1, 1).to(img.dtype)
    cx_d = cx_t.to(img.device).view(B, 1, 1).to(img.dtype)
    sh_d = sh_t.to(img.device).view(B, 1, 1).to(img.dtype)
    sw_d = sw_t.to(img.device).view(B, 1, 1).to(img.dtype)
    yy_b = yy.unsqueeze(0); xx_b = xx.unsqueeze(0)
    in_rect = ((yy_b >= cy_d) & (yy_b < cy_d + sh_d)
                & (xx_b >= cx_d) & (xx_b < cx_d + sw_d)).to(img.dtype)
    mark_d = in_rect.unsqueeze(1)  # (B, 1, H, W)
    alpha_d = alphas.to(img.device).view(B, 1, 1, 1)
    val_d = vals.to(img.device).view(B, 1, 1, 1)
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    blended = (img * (1.0 - alpha_d * mark_d.expand_as(img))
               + val_d * alpha_d * mark_d.expand_as(img)).clamp(0, 1)
    out = torch.where(gate_d.expand_as(img), blended, img)
    return out, mask


def random_text_overlay(img, mask, spec, seed_base, epoch, step):
    """Random short line strokes (handwritten notes simulation). Vectorized."""
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    nlo, nhi = spec.get("n_range", (1, 4))
    llo, lhi = spec.get("length_frac_range", (0.1, 0.3))
    thickness = float(spec.get("thickness_px", 1.5))
    alo, ahi = spec.get("alpha_range", (0.4, 0.9))
    val = float(spec.get("value", 0.1))
    n_max = int(nhi)

    gate = torch.zeros(B, dtype=torch.bool)
    alphas = torch.zeros(B, dtype=img.dtype)
    n_per_item = torch.zeros(B, dtype=torch.long)
    x0_t = torch.zeros(B, n_max, dtype=img.dtype)
    y0_t = torch.zeros(B, n_max, dtype=img.dtype)
    x1_t = torch.zeros(B, n_max, dtype=img.dtype)
    y1_t = torch.zeros(B, n_max, dtype=img.dtype)
    short_side = float(min(H, W))
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "random_text_overlay"))
        if sample_bool(p, g):
            gate[i] = True
            alphas[i] = sample_uniform(alo, ahi, g)
            n = int(torch.empty(1).uniform_(nlo, nhi + 1, generator=g).item())
            n_per_item[i] = n
            for s in range(n):
                x0 = sample_uniform(0, W - 1, g); y0 = sample_uniform(0, H - 1, g)
                length = sample_uniform(llo, lhi, g) * short_side
                ang = sample_uniform(0, math.tau, g)
                x0_t[i, s] = x0; y0_t[i, s] = y0
                x1_t[i, s] = x0 + length * math.cos(ang)
                y1_t[i, s] = y0 + length * math.sin(ang)
    x0_d = x0_t.to(img.device); y0_d = y0_t.to(img.device)
    x1_d = x1_t.to(img.device); y1_d = y1_t.to(img.device)
    nper_d = n_per_item.to(img.device)
    gate_d = gate.to(img.device)

    yy, xx = _xy_coords(H, W, img.dtype, img.device)
    yy_b = yy.unsqueeze(0); xx_b = xx.unsqueeze(0)
    strokes = torch.zeros(B, H, W, dtype=img.dtype, device=img.device)
    for s in range(n_max):
        valid = (s < nper_d).view(B, 1, 1)
        x0 = x0_d[:, s].view(B, 1, 1); y0 = y0_d[:, s].view(B, 1, 1)
        x1 = x1_d[:, s].view(B, 1, 1); y1 = y1_d[:, s].view(B, 1, 1)
        dx = x1 - x0; dy = y1 - y0
        seg_len2 = dx * dx + dy * dy + 1e-8
        t = ((xx_b - x0) * dx + (yy_b - y0) * dy) / seg_len2
        t = t.clamp(0, 1)
        px = x0 + t * dx; py = y0 + t * dy
        dist = torch.sqrt((xx_b - px) ** 2 + (yy_b - py) ** 2)
        stroke_slot = (dist < thickness).to(img.dtype)
        stroke_slot = torch.where(valid, stroke_slot, torch.zeros_like(stroke_slot))
        strokes = torch.maximum(strokes, stroke_slot)
    strokes_d = strokes.unsqueeze(1)
    alpha_d = alphas.to(img.device).view(B, 1, 1, 1)
    gate_d4 = gate_d.view(B, 1, 1, 1)
    blended = (img * (1.0 - alpha_d * strokes_d.expand_as(img))
               + val * alpha_d * strokes_d.expand_as(img)).clamp(0, 1)
    out = torch.where(gate_d4.expand_as(img), blended, img)
    return out, mask


# ─── batch 3: hue_shift (color, RGB ↔ HSV) ────────────────────────────
def _rgb_to_hsv(rgb):
    """rgb: (B, 3, H, W) float [0, 1]. Returns hsv (B, 3, H, W) with H in [0, 1)."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    mx, _ = rgb.max(dim=1, keepdim=True)
    mn, _ = rgb.min(dim=1, keepdim=True)
    delta = mx - mn
    v = mx
    s = torch.where(mx > 1e-8, delta / mx.clamp(min=1e-8), torch.zeros_like(mx))
    # Compute H per pixel based on which channel is max
    denom = delta.clamp(min=1e-8)
    h_r = ((g - b) / denom) % 6
    h_g = ((b - r) / denom) + 2
    h_b = ((r - g) / denom) + 4
    h = torch.where(mx == r, h_r, torch.where(mx == g, h_g, h_b))
    h = torch.where(delta < 1e-8, torch.zeros_like(h), h)
    h = h / 6.0  # to [0, 1)
    return torch.cat([h, s, v], dim=1)


def _hsv_to_rgb(hsv):
    """hsv: (B, 3, H, W); H in [0, 1). Returns rgb (B, 3, H, W) in [0, 1]."""
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    h6 = (h * 6.0) % 6.0
    i = torch.floor(h6)
    f = h6 - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    i = i.long() % 6
    r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p, torch.where(i == 3, p, torch.where(i == 4, t, v)))))
    gg = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v, torch.where(i == 3, q, torch.where(i == 4, p, p)))))
    bb = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t, torch.where(i == 3, v, torch.where(i == 4, v, q)))))
    return torch.cat([r, gg, bb], dim=1).clamp(0, 1)


def hue_shift(img, mask, spec, seed_base, epoch, step):
    """RGB → HSV → shift H by ±max_deg → RGB.

    spec = {"p", "max_deg", "chunk_b": int (default 2)}

    Memory: RGB↔HSV builds ~20 (B,1,H,W) intermediate tensors. For B=8 850×1100
    that's ~600 MiB peak. We chunk over the batch dim so peak ≈ chunk_b × 75 MiB
    (default chunk_b=2 → ~150 MiB peak, 4× reduction).
    """
    B = img.shape[0]
    p = float(spec.get("p", 1.0))
    max_deg = float(spec.get("max_deg", 30.0))
    chunk_b = int(spec.get("chunk_b", 2))
    gate = torch.zeros(B, dtype=torch.bool)
    shifts = torch.zeros(B, dtype=img.dtype)
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "hue_shift"))
        if sample_bool(p, g):
            gate[i] = True
            shifts[i] = sample_uniform(-max_deg / 360.0, max_deg / 360.0, g)
    shifts_d = shifts.to(img.device).view(B, 1, 1, 1)
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    out = torch.empty_like(img)
    for cs in range(0, B, chunk_b):
        ce = min(cs + chunk_b, B)
        chunk_img = img[cs:ce]
        hsv = _rgb_to_hsv(chunk_img)
        hsv[:, 0:1] = (hsv[:, 0:1] + shifts_d[cs:ce]) % 1.0
        out[cs:ce] = _hsv_to_rgb(hsv)
        del hsv, chunk_img
    out = torch.where(gate_d.expand_as(img), out, img)
    return out, mask


# ─── batch 4: conv-based ──────────────────────────────────────────────
def _gaussian_kernel_1d(sigma, dtype=torch.float32):
    """1D Gaussian kernel; returns (k,) tensor with sum 1."""
    half = int(math.ceil(3 * sigma))
    half = max(1, half)
    x = torch.arange(-half, half + 1, dtype=dtype)
    k = torch.exp(-x * x / (2 * sigma * sigma))
    return k / k.sum()


def _apply_blur_per_item(img, gate, kernels):
    """img: (B, C, H, W); kernels: list[(1, 1, kh, kw)] of per-item kernels (or None)."""
    B, C, H, W = img.shape
    out = img.clone()
    for i in range(B):
        if gate[i] and kernels[i] is not None:
            w = kernels[i].to(img.device).to(img.dtype)
            kh, kw = w.shape[-2], w.shape[-1]
            w3 = w.expand(C, 1, kh, kw)
            out[i:i + 1] = F.conv2d(img[i:i + 1], w3, padding=(kh // 2, kw // 2), groups=C)
    return out


import torch.nn.functional as F  # late-import to keep section self-contained


def gaussian_blur(img, mask, spec, seed_base, epoch, step):
    """Separable Gaussian blur per item with σ sampled from sigma_range.

    spec = {"p": prob, "sigma_range": (lo, hi)}
    """
    B = img.shape[0]
    p = float(spec.get("p", 1.0))
    smin, smax = spec.get("sigma_range", (0.5, 1.5))
    gate = torch.zeros(B, dtype=torch.bool)
    kernels = [None] * B
    for i in range(B):
        gn = cpu_generator(per_item_seed(seed_base, epoch, step, i, "gaussian_blur"))
        if sample_bool(p, gn):
            gate[i] = True
            sigma = sample_uniform(smin, smax, gn)
            k1d = _gaussian_kernel_1d(sigma)
            # Outer product → 2D kernel
            k2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)  # (kh, kw) — square
            kernels[i] = k2d.view(1, 1, *k2d.shape)
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    blurred = _apply_blur_per_item(img, gate.tolist(), kernels)
    out = torch.where(gate_d.expand_as(img), blurred, img)
    return out, mask


def motion_blur(img, mask, spec, seed_base, epoch, step):
    """Per-item 1D directional motion blur kernel (length × 1) rotated to angle.

    spec = {"p": prob, "length_range": (lo, hi) px, "angle_range_deg": (lo, hi)}
    """
    B = img.shape[0]
    p = float(spec.get("p", 1.0))
    lmin, lmax = spec.get("length_range", (5, 15))
    amin, amax = spec.get("angle_range_deg", (0, 180))
    gate = torch.zeros(B, dtype=torch.bool)
    kernels = [None] * B
    for i in range(B):
        gn = cpu_generator(per_item_seed(seed_base, epoch, step, i, "motion_blur"))
        if sample_bool(p, gn):
            gate[i] = True
            length = int(sample_uniform(lmin, lmax, gn))
            length = max(3, length | 1)  # odd
            ang_deg = sample_uniform(amin, amax, gn)
            ang = math.radians(ang_deg)
            # Draw a line on a (length, length) canvas through center, rotated by ang
            half = length // 2
            yy, xx = _xy_coords(length, length, img.dtype)
            cy, cx = float(half), float(half)
            # Distance from each cell to the line through (cx, cy) at angle ang
            sin_a, cos_a = math.sin(ang), math.cos(ang)
            # Perpendicular distance from (xx, yy) to line: |(xx-cx)*sin_a - (yy-cy)*cos_a|
            dist = ((xx - cx) * sin_a - (yy - cy) * cos_a).abs()
            # Parametric along-line position: also restrict to within length
            along = (xx - cx) * cos_a + (yy - cy) * sin_a
            mask_k = ((dist < 0.5) & (along.abs() < half + 0.5)).to(img.dtype)
            if mask_k.sum() > 0:
                kernels[i] = (mask_k / mask_k.sum()).view(1, 1, length, length)
            else:
                gate[i] = False  # safety: no valid kernel
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    blurred = _apply_blur_per_item(img, gate.tolist(), kernels)
    out = torch.where(gate_d.expand_as(img), blurred, img)
    return out, mask


def sharpness(img, mask, spec, seed_base, epoch, step):
    """Unsharp mask: out = img + α · (img − blur(img)), α sampled per item.

    spec = {"p": prob, "alpha_range": (lo, hi), "sigma": float (Gaussian σ for blur)}
    """
    B = img.shape[0]
    p = float(spec.get("p", 1.0))
    alo, ahi = spec.get("alpha_range", (0.3, 0.8))
    sigma = float(spec.get("sigma", 1.5))
    gate = torch.zeros(B, dtype=torch.bool)
    alphas = torch.zeros(B, dtype=img.dtype)
    for i in range(B):
        gn = cpu_generator(per_item_seed(seed_base, epoch, step, i, "sharpness"))
        if sample_bool(p, gn):
            gate[i] = True
            alphas[i] = sample_uniform(alo, ahi, gn)
    # Single shared blur (sigma fixed across batch) — single conv call
    k1d = _gaussian_kernel_1d(sigma).to(img.device).to(img.dtype)
    k2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)
    kh, kw = k2d.shape
    w3 = k2d.view(1, 1, kh, kw).expand(img.shape[1], 1, kh, kw)
    blur = F.conv2d(img, w3, padding=(kh // 2, kw // 2), groups=img.shape[1])
    alpha_d = alphas.to(img.device).view(B, 1, 1, 1)
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    sharpened = (img + alpha_d * (img - blur)).clamp(0, 1)
    out = torch.where(gate_d.expand_as(img), sharpened, img)
    return out, mask


def local_contrast(img, mask, spec, seed_base, epoch, step):
    """Local contrast normalization: (img − blur) / (blur + eps) → renormalised.

    spec = {"p": prob, "sigma": float, "strength_range": (lo, hi)}
    """
    B = img.shape[0]
    p = float(spec.get("p", 1.0))
    sigma = float(spec.get("sigma", 5.0))
    slo, shi = spec.get("strength_range", (0.3, 0.7))
    gate = torch.zeros(B, dtype=torch.bool)
    strengths = torch.zeros(B, dtype=img.dtype)
    for i in range(B):
        gn = cpu_generator(per_item_seed(seed_base, epoch, step, i, "local_contrast"))
        if sample_bool(p, gn):
            gate[i] = True
            strengths[i] = sample_uniform(slo, shi, gn)
    k1d = _gaussian_kernel_1d(sigma).to(img.device).to(img.dtype)
    k2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)
    kh, kw = k2d.shape
    w3 = k2d.view(1, 1, kh, kw).expand(img.shape[1], 1, kh, kw)
    blur = F.conv2d(img, w3, padding=(kh // 2, kw // 2), groups=img.shape[1])
    s_d = strengths.to(img.device).view(B, 1, 1, 1)
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    enhanced = (img + s_d * (img - blur) / (blur + 0.1)).clamp(0, 1)
    out = torch.where(gate_d.expand_as(img), enhanced, img)
    return out, mask


def jpeg_approx(img, mask, spec, seed_base, epoch, step):
    """JPEG-like compression approximation: additive noise + 3×3 box blur.
    (libjpeg DCT is non-trivial to make bit-exact on GPU; this approximation
    captures the visual character — high-freq attenuation + quantization noise.)

    spec = {"p": prob, "noise_sigma_range": (lo, hi)}
    """
    B = img.shape[0]
    p = float(spec.get("p", 1.0))
    smin, smax = spec.get("noise_sigma_range", (0.005, 0.02))
    gate = torch.zeros(B, dtype=torch.bool)
    noise = torch.zeros_like(img.cpu()) if img.is_cuda else torch.zeros_like(img)
    for i in range(B):
        gn = cpu_generator(per_item_seed(seed_base, epoch, step, i, "jpeg_approx"))
        if sample_bool(p, gn):
            gate[i] = True
            sigma = sample_uniform(smin, smax, gn)
            noise[i] = torch.empty_like(img[0].cpu() if img.is_cuda else img[0]).normal_(0.0, 1.0, generator=gn) * sigma
    noise_d = noise.to(img.device)
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    # 3×3 box blur shared across batch
    kbox = torch.full((1, 1, 3, 3), 1.0 / 9.0, dtype=img.dtype, device=img.device)
    w3 = kbox.expand(img.shape[1], 1, 3, 3)
    blurred = F.conv2d(img + noise_d, w3, padding=1, groups=img.shape[1]).clamp(0, 1)
    out = torch.where(gate_d.expand_as(img), blurred, img)
    return out, mask


# ─── batch 5: paper / content overlays ────────────────────────────────
def stains(img, mask, spec, seed_base, epoch, step):
    """Soft elliptical blobs blended toward a stain colour. Vectorized over B."""
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    nlo, nhi = spec.get("n_range", (1, 5))
    rlo, rhi = spec.get("radius_frac_range", (0.05, 0.2))
    alo, ahi = spec.get("alpha_range", (0.2, 0.5))
    vlo, vhi = spec.get("value_range", (0.3, 0.7))
    n_max = int(nhi)

    gate = torch.zeros(B, dtype=torch.bool)
    alpha_acc = torch.zeros(B, dtype=img.dtype)
    val_acc = torch.zeros(B, dtype=img.dtype)
    n_per_item = torch.zeros(B, dtype=torch.long)
    cy_t = torch.zeros(B, n_max, dtype=img.dtype)
    cx_t = torch.zeros(B, n_max, dtype=img.dtype)
    rx_t = torch.ones(B, n_max, dtype=img.dtype)
    ry_t = torch.ones(B, n_max, dtype=img.dtype)
    ang_t = torch.zeros(B, n_max, dtype=img.dtype)
    short = float(min(H, W))
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "stains"))
        if sample_bool(p, g):
            gate[i] = True
            n = int(torch.empty(1).uniform_(nlo, nhi + 1, generator=g).item())
            n_per_item[i] = n
            alpha_acc[i] = sample_uniform(alo, ahi, g)
            val_acc[i] = sample_uniform(vlo, vhi, g)
            for s in range(n):
                cy_t[i, s] = sample_uniform(0, H - 1, g)
                cx_t[i, s] = sample_uniform(0, W - 1, g)
                rx_t[i, s] = sample_uniform(rlo, rhi, g) * short
                ry_t[i, s] = sample_uniform(rlo, rhi, g) * short
                ang_t[i, s] = sample_uniform(0, math.pi, g)
    cy_d = cy_t.to(img.device); cx_d = cx_t.to(img.device)
    rx_d = rx_t.to(img.device); ry_d = ry_t.to(img.device)
    ang_d = ang_t.to(img.device)
    nper_d = n_per_item.to(img.device)
    gate_d = gate.to(img.device)

    yy, xx = _xy_coords(H, W, img.dtype, img.device)
    yy_b = yy.unsqueeze(0); xx_b = xx.unsqueeze(0)
    blob = torch.zeros(B, H, W, dtype=img.dtype, device=img.device)
    for s in range(n_max):
        valid = (s < nper_d).view(B, 1, 1)
        cy = cy_d[:, s].view(B, 1, 1); cx = cx_d[:, s].view(B, 1, 1)
        rx = rx_d[:, s].view(B, 1, 1); ry = ry_d[:, s].view(B, 1, 1)
        ca = torch.cos(ang_d[:, s]).view(B, 1, 1)
        sa = torch.sin(ang_d[:, s]).view(B, 1, 1)
        dx_ = xx_b - cx; dy_ = yy_b - cy
        ex = ca * dx_ + sa * dy_
        ey = -sa * dx_ + ca * dy_
        ell = torch.exp(-((ex / rx) ** 2 + (ey / ry) ** 2))
        ell = torch.where(valid, ell, torch.zeros_like(ell))
        blob = torch.maximum(blob, ell)
    blob_d = blob.unsqueeze(1)
    alpha_d = alpha_acc.to(img.device).view(B, 1, 1, 1)
    val_d = val_acc.to(img.device).view(B, 1, 1, 1)
    gate_d4 = gate_d.view(B, 1, 1, 1)
    bmask = (blob_d * alpha_d).expand_as(img)
    blended = (img * (1.0 - bmask) + val_d * bmask).clamp(0, 1)
    out = torch.where(gate_d4.expand_as(img), blended, img)
    return out, mask


def creases(img, mask, spec, seed_base, epoch, step):
    """Linear shadow stripes (paper creases). Each crease is a line with
    exponential falloff perpendicular distance, blended as darkening.

    spec = {"p": prob, "n_range": (lo, hi), "width_px": float,
            "darkness_range": (lo, hi)}
    """
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    nlo, nhi = spec.get("n_range", (1, 3))
    width = float(spec.get("width_px", 6.0))
    dlo, dhi = spec.get("darkness_range", (0.1, 0.35))
    gate = torch.zeros(B, dtype=torch.bool)
    shade = torch.zeros(B, 1, H, W, dtype=img.dtype)
    dark_acc = torch.zeros(B, dtype=img.dtype)
    yy, xx = _xy_coords(H, W, img.dtype)
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "creases"))
        if sample_bool(p, g):
            gate[i] = True
            dark_acc[i] = sample_uniform(dlo, dhi, g)
            n = int(torch.empty(1).uniform_(nlo, nhi + 1, generator=g).item())
            for _ in range(n):
                cx = sample_uniform(0, W - 1, g)
                cy = sample_uniform(0, H - 1, g)
                ang = sample_uniform(0, math.pi, g)
                sin_a, cos_a = math.sin(ang), math.cos(ang)
                dist = ((xx - cx) * sin_a - (yy - cy) * cos_a).abs()
                line = torch.exp(-(dist * dist) / (2 * width * width))
                shade[i, 0] = torch.maximum(shade[i, 0], line)
    shade_d = shade.to(img.device)
    dark_d = dark_acc.to(img.device).view(B, 1, 1, 1)
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    factor = (1.0 - dark_d * shade_d).expand_as(img).clamp(0, 1)
    new_img = (img * factor).clamp(0, 1)
    out = torch.where(gate_d.expand_as(img), new_img, img)
    return out, mask


def specular_highlight(img, mask, spec, seed_base, epoch, step):
    """1-3 large Gaussian bright spots. Vectorized over B."""
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    nlo, nhi = spec.get("n_range", (1, 3))
    slo, shi = spec.get("sigma_frac_range", (0.05, 0.15))
    ilo, ihi = spec.get("intensity_range", (0.3, 0.7))
    n_max = int(nhi)
    gate = torch.zeros(B, dtype=torch.bool)
    n_per_item = torch.zeros(B, dtype=torch.long)
    cy_t = torch.zeros(B, n_max, dtype=img.dtype)
    cx_t = torch.zeros(B, n_max, dtype=img.dtype)
    sig_t = torch.ones(B, n_max, dtype=img.dtype)
    int_t = torch.zeros(B, n_max, dtype=img.dtype)
    short = float(min(H, W))
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "specular_highlight"))
        if sample_bool(p, g):
            gate[i] = True
            n = int(torch.empty(1).uniform_(nlo, nhi + 1, generator=g).item())
            n_per_item[i] = n
            for s in range(n):
                cy_t[i, s] = sample_uniform(0, H - 1, g)
                cx_t[i, s] = sample_uniform(0, W - 1, g)
                sig_t[i, s] = sample_uniform(slo, shi, g) * short
                int_t[i, s] = sample_uniform(ilo, ihi, g)
    cy_d = cy_t.to(img.device); cx_d = cx_t.to(img.device)
    sig_d = sig_t.to(img.device); int_d = int_t.to(img.device)
    nper_d = n_per_item.to(img.device); gate_d = gate.to(img.device)
    yy, xx = _xy_coords(H, W, img.dtype, img.device)
    yy_b = yy.unsqueeze(0); xx_b = xx.unsqueeze(0)
    add = torch.zeros(B, H, W, dtype=img.dtype, device=img.device)
    for s in range(n_max):
        valid = (s < nper_d).view(B, 1, 1)
        cy = cy_d[:, s].view(B, 1, 1); cx = cx_d[:, s].view(B, 1, 1)
        sig = sig_d[:, s].view(B, 1, 1); i_s = int_d[:, s].view(B, 1, 1)
        gaus = torch.exp(-((xx_b - cx) ** 2 + (yy_b - cy) ** 2) / (2 * sig * sig)) * i_s
        gaus = torch.where(valid, gaus, torch.zeros_like(gaus))
        add = torch.maximum(add, gaus)
    add_d = add.unsqueeze(1)
    gate_d4 = gate_d.view(B, 1, 1, 1)
    new_img = (img + add_d.expand_as(img)).clamp(0, 1)
    out = torch.where(gate_d4.expand_as(img), new_img, img)
    return out, mask


def specular_streaks(img, mask, spec, seed_base, epoch, step):
    """1-3 long line-shaped bright reflections. Vectorized over B."""
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    nlo, nhi = spec.get("n_range", (1, 3))
    llo, lhi = spec.get("length_frac_range", (0.2, 0.6))
    thickness = float(spec.get("thickness_px", 4.0))
    ilo, ihi = spec.get("intensity_range", (0.3, 0.6))
    n_max = int(nhi)
    gate = torch.zeros(B, dtype=torch.bool)
    n_per_item = torch.zeros(B, dtype=torch.long)
    x0_t = torch.zeros(B, n_max, dtype=img.dtype)
    y0_t = torch.zeros(B, n_max, dtype=img.dtype)
    x1_t = torch.zeros(B, n_max, dtype=img.dtype)
    y1_t = torch.zeros(B, n_max, dtype=img.dtype)
    int_t = torch.zeros(B, n_max, dtype=img.dtype)
    short = float(min(H, W))
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "specular_streaks"))
        if sample_bool(p, g):
            gate[i] = True
            n = int(torch.empty(1).uniform_(nlo, nhi + 1, generator=g).item())
            n_per_item[i] = n
            for s in range(n):
                x0 = sample_uniform(0, W - 1, g); y0 = sample_uniform(0, H - 1, g)
                length = sample_uniform(llo, lhi, g) * short
                ang = sample_uniform(0, math.tau, g)
                x0_t[i, s] = x0; y0_t[i, s] = y0
                x1_t[i, s] = x0 + length * math.cos(ang)
                y1_t[i, s] = y0 + length * math.sin(ang)
                int_t[i, s] = sample_uniform(ilo, ihi, g)
    x0_d = x0_t.to(img.device); y0_d = y0_t.to(img.device)
    x1_d = x1_t.to(img.device); y1_d = y1_t.to(img.device)
    int_d = int_t.to(img.device); nper_d = n_per_item.to(img.device)
    gate_d = gate.to(img.device)
    yy, xx = _xy_coords(H, W, img.dtype, img.device)
    yy_b = yy.unsqueeze(0); xx_b = xx.unsqueeze(0)
    add = torch.zeros(B, H, W, dtype=img.dtype, device=img.device)
    for s in range(n_max):
        valid = (s < nper_d).view(B, 1, 1)
        x0 = x0_d[:, s].view(B, 1, 1); y0 = y0_d[:, s].view(B, 1, 1)
        x1 = x1_d[:, s].view(B, 1, 1); y1 = y1_d[:, s].view(B, 1, 1)
        dx = x1 - x0; dy = y1 - y0
        seg_len2 = dx * dx + dy * dy + 1e-8
        t = (((xx_b - x0) * dx + (yy_b - y0) * dy) / seg_len2).clamp(0, 1)
        px = x0 + t * dx; py = y0 + t * dy
        dist = torch.sqrt((xx_b - px) ** 2 + (yy_b - py) ** 2)
        falloff = torch.exp(-dist * dist / (2 * thickness * thickness)) * int_d[:, s].view(B, 1, 1)
        falloff = torch.where(valid, falloff, torch.zeros_like(falloff))
        add = torch.maximum(add, falloff)
    add_d = add.unsqueeze(1)
    gate_d4 = gate_d.view(B, 1, 1, 1)
    new_img = (img + add_d.expand_as(img)).clamp(0, 1)
    out = torch.where(gate_d4.expand_as(img), new_img, img)
    return out, mask


# ─── batch 6: complex (CLAHE + paper_texture_overlay) ────────────────
def clahe(img, mask, spec, seed_base, epoch, step):
    """Contrast Limited Adaptive Histogram Equalization (per-tile hist EQ),
    pure torch. tile_size matches cv2.createCLAHE default 8×8.

    spec = {"p": prob, "clip_limit": float (clip per tile bin), "tile_size": int,
            "n_bins": int}

    Bit-exact tolerance: histogram bin assignment via floor() can flip at exact
    bin boundaries between CPU/CUDA fp32 rounding. atol class is 1e-4 (conv-style);
    if observed > 1e-4, relax to 1e-3 in TOLERANCE map with docstring note.
    """
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    clip_limit = float(spec.get("clip_limit", 2.0))
    tile_size = int(spec.get("tile_size", 8))
    n_bins = int(spec.get("n_bins", 256))
    gate = torch.zeros(B, dtype=torch.bool)
    for i in range(B):
        gn = cpu_generator(per_item_seed(seed_base, epoch, step, i, "clahe"))
        if sample_bool(p, gn):
            gate[i] = True
    gate_d = gate.to(img.device).view(B, 1, 1, 1)

    # Chunk over Bx (B·C) to cap VRAM peak. With B=8, C=3 → Bx=24, default
    # chunk_bx=8 → per-chunk peak ≈ 1/3 of monolithic (was 1.7 GB → ~580 MiB,
    # below the 500 MB target with breathing room from accumulated frees).
    Bx = B * C
    chunk_bx = int(spec.get("chunk_bx", 8))
    flat = img.reshape(Bx, H, W)
    # Tile dimensions — ceil so Hp/Wp >= H/W (pad is non-negative)
    th = max(1, (H + tile_size - 1) // tile_size)
    tw = max(1, (W + tile_size - 1) // tile_size)
    Hp = th * tile_size; Wp = tw * tile_size
    pad_h = Hp - H; pad_w = Wp - W
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=img.device, dtype=img.dtype)
    tile_area = tile_size * tile_size
    clip = clip_limit * tile_area / n_bins

    eq = torch.empty_like(flat)
    for cstart in range(0, Bx, chunk_bx):
        cend = min(cstart + chunk_bx, Bx)
        chunk = flat[cstart:cend]
        if pad_h > 0 or pad_w > 0:
            chunk = F.pad(chunk.unsqueeze(1), (0, pad_w, 0, pad_h), mode="reflect").squeeze(1)
        bxc = cend - cstart
        tiled = chunk.view(bxc, th, tile_size, tw, tile_size).permute(0, 1, 3, 2, 4).contiguous()
        bin_idx = torch.bucketize(tiled.clamp(0, 1), boundaries).clamp(0, n_bins - 1)
        flat_bins = bin_idx.reshape(bxc, th, tw, -1)
        counts = torch.zeros(bxc, th, tw, n_bins, dtype=img.dtype, device=img.device)
        counts.scatter_add_(-1, flat_bins, torch.ones_like(flat_bins, dtype=img.dtype))
        excess = (counts - clip).clamp(min=0).sum(dim=-1, keepdim=True)
        counts_clipped = counts.clamp(max=clip) + excess / n_bins
        del counts, excess
        cdf = torch.cumsum(counts_clipped, dim=-1)
        cdf = cdf / cdf[..., -1:].clamp(min=1e-8)
        cdf_flat = cdf.view(bxc * th * tw, n_bins)
        bin_flat = bin_idx.reshape(bxc * th * tw, tile_size * tile_size)
        looked_flat = torch.gather(cdf_flat, 1, bin_flat)
        del cdf, cdf_flat, bin_flat, bin_idx, counts_clipped
        looked = looked_flat.view(bxc, th, tw, tile_size, tile_size)
        out_padded = looked.permute(0, 1, 3, 2, 4).contiguous().view(bxc, Hp, Wp)
        eq[cstart:cend] = out_padded[..., :H, :W]
        del looked, looked_flat, out_padded, chunk
    eq = eq.reshape(B, C, H, W)
    out = torch.where(gate_d.expand_as(img), eq, img)
    return out, mask


def paper_texture_overlay(img, mask, spec, seed_base, epoch, step):
    """Multiplicative procedural texture overlay simulating paper grain.

    spec = {"p": prob,
            "blend_strength": float (overlay magnitude, default 0.3),
            "sigma": float (low-res cell size for the texture, default 30)}

    Generates texture per item by sampling a low-res Gaussian random field
    (size ≈ (H/sigma, W/sigma)), bilinear-upsampling to (H, W), z-normalising
    it to a centred multiplicative factor in [1 - strength, 1 + strength],
    then multiplying the image.
    """
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    strength = float(spec.get("blend_strength", 0.3))
    sigma = float(spec.get("sigma", 30.0))
    gate = torch.zeros(B, dtype=torch.bool)
    texture = torch.ones(B, 1, H, W, dtype=img.dtype)
    for i in range(B):
        gn = cpu_generator(per_item_seed(seed_base, epoch, step, i, "paper_texture_overlay"))
        if sample_bool(p, gn):
            gate[i] = True
            # Low-res random → bilinear upsample (Gaussian-like smoothness)
            hl = max(4, int(H / sigma)); wl = max(4, int(W / sigma))
            raw = torch.empty(1, 1, hl, wl).normal_(0.0, 1.0, generator=gn)
            up = F.interpolate(raw, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            # Normalise to [0.85, 1.15] (multiplicative factor centered at 1)
            up = (up - up.mean()) / (up.std() + 1e-6)
            texture[i, 0] = (1.0 + 0.15 * up).clamp(1.0 - strength, 1.0 + strength)
    texture_d = texture.to(img.device)
    gate_d = gate.to(img.device).view(B, 1, 1, 1)
    new_img = (img * texture_d.expand_as(img)).clamp(0, 1)
    out = torch.where(gate_d.expand_as(img), new_img, img)
    return out, mask


def background_compose(img, mask, spec, seed_base, epoch, step):
    """Compose the paper image onto a random background outside the mask.

        composed = img * mask + bg * (1 - mask)

    Background source per item:
      - With probability `p_dtd`, draw from a DTD texture cache (a stacked
        numpy `.npy` of pre-cropped textures). If `dtd_cache_path` is not
        provided or the file doesn't exist, falls back to procedural noise.
      - Otherwise generate procedural noise: low-resolution Gaussian random
        bilinear-upsampled to (H, W), normalised, multiplied by a per-item
        brightness factor in `bg_brightness_range`.

    spec = {"p": prob,
            "dtd_cache_path": Optional[str] (path to .npy texture bank),
            "p_dtd": float (prob of pulling from DTD cache vs procedural noise),
            "bg_brightness_range": (lo, hi),
            "sigma": float (smoothness for procedural noise),
            "channel_jitter": float (per-channel multiplier variance)}

    The paper_mask is interpreted as 1.0 = paper interior, 0.0 = outside (the
    region that gets replaced by `bg`).
    """
    B, C, H, W = img.shape
    p = float(spec.get("p", 1.0))
    p_dtd = float(spec.get("p_dtd", 0.5))
    blo, bhi = spec.get("bg_brightness_range", (0.6, 1.0))
    sigma = float(spec.get("sigma", 40.0))
    ch_jitter = float(spec.get("channel_jitter", 0.1))
    dtd_cache_path = spec.get("dtd_cache_path", None)

    # Try to load DTD cache once (per-process, cached on the spec dict for
    # subsequent calls to skip the IO).
    dtd_textures = spec.get("_dtd_textures_cached", None)
    if dtd_textures is None and dtd_cache_path:
        from pathlib import Path as _P
        import numpy as _np
        cp = _P(dtd_cache_path).expanduser()
        if cp.exists():
            arr = _np.load(str(cp))
            # Expected shape: (N, 3, h, w) or (N, h, w, 3) — accept either
            t = torch.from_numpy(arr)
            if t.dim() == 4 and t.shape[-1] == 3:
                t = t.permute(0, 3, 1, 2).contiguous()
            dtd_textures = t.float() / (255.0 if t.dtype == torch.uint8 else 1.0)
            spec["_dtd_textures_cached"] = dtd_textures

    if mask is None:
        # Nothing to compose against — pass-through. Composing onto an all-mask
        # area degenerates to img unchanged.
        return img, mask

    # Per-item RNG sample of: gate, p_dtd choice, brightness, channel jitter,
    # procedural seed (per-pixel via low-res random tensor allocation seed),
    # DTD index (if available).
    gate = torch.zeros(B, dtype=torch.bool)
    use_dtd = torch.zeros(B, dtype=torch.bool)
    bright = torch.ones(B, dtype=img.dtype)
    ch_mult = torch.ones(B, 3, dtype=img.dtype)
    dtd_idx = torch.zeros(B, dtype=torch.long)
    # Pre-sample low-resolution procedural noise on CPU (bit-exact RNG)
    hl = max(4, int(H / sigma)); wl = max(4, int(W / sigma))
    proc_low = torch.zeros(B, 3, hl, wl, dtype=img.dtype)
    n_dtd = dtd_textures.shape[0] if dtd_textures is not None else 0
    for i in range(B):
        g = cpu_generator(per_item_seed(seed_base, epoch, step, i, "background_compose"))
        if not sample_bool(p, g):
            continue
        gate[i] = True
        bright[i] = sample_uniform(blo, bhi, g)
        for c in range(3):
            ch_mult[i, c] = 1.0 + sample_uniform(-ch_jitter, ch_jitter, g)
        if n_dtd > 0 and sample_bool(p_dtd, g):
            use_dtd[i] = True
            dtd_idx[i] = int(torch.empty(1).uniform_(0, n_dtd, generator=g).item())
        # Procedural noise always sampled to keep RNG state deterministic, even
        # when DTD is used for this item (so the bit-exact sequence is stable
        # regardless of which branch wins).
        proc_low[i] = torch.empty(3, hl, wl).normal_(0.0, 1.0, generator=g)

    # Move to device
    gate_d = gate.to(img.device)
    use_dtd_d = use_dtd.to(img.device)
    bright_d = bright.to(img.device).view(B, 1, 1, 1)
    ch_mult_d = ch_mult.to(img.device).view(B, 3, 1, 1)
    proc_low_d = proc_low.to(img.device)

    # Procedural bg: upsample low-res random to (H, W), normalise to [0.5±0.3]
    proc_full = F.interpolate(proc_low_d, size=(H, W),
                               mode="bilinear", align_corners=False)
    pm = proc_full.mean(dim=(2, 3), keepdim=True)
    ps = proc_full.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    proc_full = ((proc_full - pm) / ps * 0.15 + 0.5).clamp(0, 1)
    bg = proc_full * ch_mult_d * bright_d

    # DTD bg: pull selected texture per item, resize to (H, W)
    if n_dtd > 0 and use_dtd.any().item():
        dtd_d = dtd_textures.to(img.device)
        for i in range(B):
            if use_dtd[i]:
                tex = dtd_d[int(dtd_idx[i])]
                if tex.shape[-2:] != (H, W):
                    tex = F.interpolate(tex.unsqueeze(0), size=(H, W),
                                         mode="bilinear", align_corners=False)[0]
                bg[i] = (tex * ch_mult_d[i] * bright_d[i, 0]).clamp(0, 1)

    composed = img * mask + bg * (1.0 - mask)
    composed = composed.clamp(0, 1)
    out = torch.where(gate_d.view(B, 1, 1, 1).expand_as(img), composed, img)
    return out, mask


PHOTOMETRIC_PRIMITIVES = {
    "gamma": gamma,
    "gaussian_noise": gaussian_noise,
    "salt_pepper_noise": salt_pepper_noise,
    "color_jitter": color_jitter,
    "random_grayscale": random_grayscale,
    "cutout": cutout,
    "lighting": lighting,
    "vignette": vignette,
    "salt_patches": salt_patches,
    "watermark": watermark,
    "random_text_overlay": random_text_overlay,
    "hue_shift": hue_shift,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "sharpness": sharpness,
    "local_contrast": local_contrast,
    "jpeg_approx": jpeg_approx,
    "stains": stains,
    "creases": creases,
    "specular_highlight": specular_highlight,
    "specular_streaks": specular_streaks,
    "clahe": clahe,
    "paper_texture_overlay": paper_texture_overlay,
    "background_compose": background_compose,
}
