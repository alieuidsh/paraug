"""Cross-device deterministic RNG + global determinism toggles for paraug.

The pipeline's bit-exact guarantee relies on **CPU-side random number generation**
even when the image tensors live on GPU. CUDA RNG (cuRAND) and CPU RNG (MT19937)
emit different sequences from the same seed, so any primitive that calls
`torch.rand(..., device=cuda)` would diverge from its CPU counterpart at bit level.
Instead, every primitive samples its decisions on CPU via `cpu_generator(seed)`,
then `.to(img.device)` to feed the actual op. Standard torch ops on those
samples are deterministic across CPU/CUDA within a tight tolerance (see
`tests/test_parity.py` for the verified bound: 1e-6 elementwise, 2e-4 for
grid_sample- and conv-class ops).
"""
import torch


# Salt namespace for primitives — each primitive gets a unique salt so that its
# per-item RNG stream does not collide with any other primitive in the pipeline.
SALT = {
    # Geometric
    "tps":               1,
    "affine":            2,
    "perspective":       3,
    "random_crop_pad":   4,
    "elastic_transform": 5,
    "optical_distortion": 6,
    "random_shadow":     7,
    # Photometric
    "gaussian_blur":     11,
    "motion_blur":       12,
    "vignette":          13,
    "specular_highlight": 14,
    "specular_streaks":  15,
    "gamma":             16,
    "jpeg_approx":       17,
    "gaussian_noise":    18,
    # Color
    "color_jitter":      21,
    "hue_shift":         22,
    "random_grayscale":  23,
    "lighting":          24,
    # Paper / content
    "stains":            31,
    "creases":           32,
    "cutout":            33,
    "salt_patches":      34,
    "paper_texture_overlay": 35,
    "watermark":         36,
    "random_text_overlay": 37,
    "background_compose": 38,
    # Contrast / sharpness
    "sharpness":         41,
    "clahe":             42,
    "local_contrast":    43,
    "salt_pepper_noise": 44,
}


def per_item_seed(seed_base: int, epoch: int, step: int,
                  item_i: int, primitive: str) -> int:
    """Deterministic per-item seed: seed_base + epoch*1e6 + step*1024 + i + primitive_salt*1e9.

    `primitive_salt * 1e9` puts each primitive on its own seed range so two
    primitives applied to the same item do not share an RNG state.
    """
    s = SALT.get(primitive, 0)
    return (int(seed_base)
            + int(epoch) * 1_000_000
            + int(step) * 1024
            + int(item_i)
            + s * 1_000_000_000)


def cpu_generator(seed: int) -> torch.Generator:
    """Return a CPU-side `torch.Generator` seeded with `seed`. The pipeline always
    samples randomness on CPU first to guarantee bit-exact equivalence across
    CPU and CUDA back-ends (cuRAND ≠ MT19937 even at fixed seed)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def sample_uniform(low: float, high: float, generator: torch.Generator) -> float:
    """Single scalar uniform sample using the given (CPU) generator."""
    return float(torch.empty(1).uniform_(float(low), float(high), generator=generator))


def sample_bool(p: float, generator: torch.Generator) -> bool:
    """Bernoulli sample: True with probability p, False with 1-p."""
    return bool(torch.rand(1, generator=generator).item() < p)


def set_deterministic(state: bool = True, warn_only: bool = True) -> None:
    """Toggle global deterministic mode. Call once per process before any aug.

    `warn_only=True` keeps non-deterministic kernels usable (with a warning),
    which is required for some grid_sample paths. Set False if you need a hard
    failure when a non-deterministic op is invoked.
    """
    torch.use_deterministic_algorithms(state, warn_only=warn_only)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = state
        torch.backends.cudnn.benchmark = not state
