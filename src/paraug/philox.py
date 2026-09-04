"""Device-side counter-based RNG (Philox4x32-10) implemented in pure torch int64 ops.

Why: paraug's bit-exact CPU/CUDA guarantee originally forced every random
field to be sampled on CPU (MT19937) and copied to the device. For dense
per-pixel fields (gaussian_noise, jpeg_approx, salt_pepper_noise, …) that is
the dominant cost — a 4090 ends up no faster than a 3060 because the GPU is
waiting on single-threaded CPU RNG + a 24 MB H2D copy per batch.

Philox is a *counter-based* generator: output = f(key, counter), no state.
Implemented here with integer arithmetic only (32-bit values held in int64,
multiplication split into 16-bit halves so no intermediate exceeds 2**40),
so the uint32 stream is **bit-identical** on CPU and CUDA. Converting to
float uniform is an exact operation ((x >> 8) * 2**-24); the Box–Muller
normal transform then goes through sqrt/log/cos which may differ by ~1 ulp
across back-ends — that is already inside paraug's elementwise tolerance
(atol 1e-6) once multiplied by a noise sigma.

Per-item determinism is preserved: item `i` of a batch uses counter word
c2 = i and key = f(seed), so its field does not depend on batch size or on
which other items share the batch (same contract as the CPU path).

Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3", SC'11.
"""
import math
import torch

M32 = 0xFFFFFFFF
_M0 = 0xD2511F53
_M1 = 0xCD9E8D57
_W0 = 0x9E3779B9
_W1 = 0xBB67AE85


def _mulhilo32(a: int, b: torch.Tensor):
    """(hi, lo) of the 64-bit product a*b for a scalar a < 2**32 and int64 tensor b < 2**32.
    Split into 16-bit halves so every intermediate stays < 2**40 (no overflow)."""
    a1, a0 = a >> 16, a & 0xFFFF
    b1, b0 = b >> 16, b & 0xFFFF
    ll = a0 * b0
    mid = a0 * b1 + a1 * b0
    hh = a1 * b1
    lo64 = ll + ((mid & 0xFFFF) << 16)
    lo = lo64 & M32
    hi = (hh + (mid >> 16) + (lo64 >> 32)) & M32
    return hi, lo


def philox4x32_10(c0, c1, c2, c3, k0: int, k1: int):
    """One Philox4x32-10 block. Counters are int64 tensors holding uint32 values
    (broadcastable), keys are Python ints < 2**32. Returns 4 int64 tensors of uint32."""
    for r in range(10):
        hi0, lo0 = _mulhilo32(_M0, c0)
        hi1, lo1 = _mulhilo32(_M1, c2)
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0), lo1, (hi0 ^ c3 ^ k1), lo0
        if r < 9:
            k0 = (k0 + _W0) & M32
            k1 = (k1 + _W1) & M32
    return c0, c1, c2, c3


def _key(seed: int):
    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    return seed & M32, (seed >> 32) & M32


def philox_uint32(seed: int, n: int, device, stream: int = 0, offset: int = 0) -> torch.Tensor:
    """`n` uint32 values (as int64 tensor) for (seed, stream), starting at counter `offset`.
    Equivalent across devices bit-for-bit."""
    k0, k1 = _key(seed)
    nblk = (n + 3) // 4
    idx = torch.arange(nblk, device=device, dtype=torch.int64) + (offset // 4)
    c0 = idx & M32
    c1 = (idx >> 32) & M32
    c2 = torch.full_like(idx, int(stream) & M32)
    c3 = torch.zeros_like(idx)
    o = philox4x32_10(c0, c1, c2, c3, k0, k1)
    return torch.stack(o, dim=1).reshape(-1)[:n]


def philox_uniform(seed: int, shape, device, dtype=torch.float32, stream: int = 0) -> torch.Tensor:
    """Uniform [0, 1) of `shape`. Exact conversion: 24 high bits -> float32 multiple of 2**-24."""
    n = math.prod(shape)
    u = philox_uint32(seed, n, device, stream)
    return ((u >> 8).to(torch.float32) * (1.0 / 16777216.0)).to(dtype).reshape(shape)


def philox_normal(seed: int, shape, device, dtype=torch.float32, stream: int = 0) -> torch.Tensor:
    """Standard normal of `shape` via Box–Muller on Philox uniforms."""
    n = math.prod(shape)
    npair = (n + 1) // 2
    u = philox_uint32(seed, 2 * npair, device, stream)
    u = (u >> 8).to(torch.float32) * (1.0 / 16777216.0)
    u1 = 1.0 - u[:npair]            # (0, 1]  -> log is finite
    u2 = u[npair:]
    r = torch.sqrt(-2.0 * torch.log(u1))
    theta = (2.0 * math.pi) * u2
    z = torch.cat([r * torch.cos(theta), r * torch.sin(theta)])[:n]
    return z.to(dtype).reshape(shape)


def philox_uniform_batch(seed_fn, B: int, item_shape, device, dtype=torch.float32) -> torch.Tensor:
    """(B, *item_shape) uniforms where item i uses seed_fn(i) as key and stream i.
    Vectorised: one Philox pass over the whole batch, per-item independence kept via
    per-item key + counter stream (no Python loop over pixels)."""
    n = math.prod(item_shape)
    nblk = (n + 3) // 4
    idx = torch.arange(nblk, device=device, dtype=torch.int64)
    c0 = (idx & M32).unsqueeze(0).expand(B, nblk)
    c1 = ((idx >> 32) & M32).unsqueeze(0).expand(B, nblk)
    c2 = torch.arange(B, device=device, dtype=torch.int64).unsqueeze(1).expand(B, nblk)
    c3 = torch.zeros(B, nblk, device=device, dtype=torch.int64)
    keys = [_key(seed_fn(i)) for i in range(B)]
    k0 = torch.tensor([k[0] for k in keys], device=device, dtype=torch.int64).unsqueeze(1)
    k1 = torch.tensor([k[1] for k in keys], device=device, dtype=torch.int64).unsqueeze(1)
    o = _philox_tensor_key(c0, c1, c2, c3, k0, k1)
    u = torch.stack(o, dim=2).reshape(B, -1)[:, :n]
    return ((u >> 8).to(torch.float32) * (1.0 / 16777216.0)).to(dtype).reshape(B, *item_shape)


def philox_normal_batch(seed_fn, B: int, item_shape, device, dtype=torch.float32) -> torch.Tensor:
    """(B, *item_shape) standard normals, per-item keyed like philox_uniform_batch."""
    n = math.prod(item_shape)
    npair = (n + 1) // 2
    u = philox_uniform_batch(seed_fn, B, (2 * npair,), device, torch.float32)
    u1 = 1.0 - u[:, :npair]
    u2 = u[:, npair:]
    r = torch.sqrt(-2.0 * torch.log(u1))
    theta = (2.0 * math.pi) * u2
    z = torch.cat([r * torch.cos(theta), r * torch.sin(theta)], dim=1)[:, :n]
    return z.to(dtype).reshape(B, *item_shape)


def _hash_uniform_batch(seed_fn, B: int, item_shape, device, dtype=torch.float32) -> torch.Tensor:
    """Cheaper alternative to Philox for augmentation-grade randomness: lowbias32
    integer hash (Wellons) of (per-item key ^ index). ~2.5x less memory traffic
    than Philox4x32-10 in unfused torch ops (matters on bandwidth-limited GPUs).
    Same guarantees: integer-only -> bit-identical across CPU/CUDA; per-item
    keyed -> independent of batch size. Not crush-tested; fine for noise fields."""
    n = math.prod(item_shape)
    idx = torch.arange(n, device=device, dtype=torch.int64).unsqueeze(0)          # (1, n)
    keys = torch.tensor([(_key(seed_fn(i))[0] ^ (_key(seed_fn(i))[1] * 0x9E3779B1 & M32))
                         for i in range(B)], device=device, dtype=torch.int64).unsqueeze(1)
    x = (idx ^ keys) & M32
    x ^= x >> 16; x = _mulhilo32(0x7FEB352D, x)[1]
    x ^= x >> 15; x = _mulhilo32(0x846CA68B, x)[1]
    x ^= x >> 16
    return ((x >> 8).to(torch.float32) * (1.0 / 16777216.0)).to(dtype).reshape(B, *item_shape)


def _hash_normal_batch(seed_fn, B: int, item_shape, device, dtype=torch.float32) -> torch.Tensor:
    n = math.prod(item_shape)
    npair = (n + 1) // 2
    u = _hash_uniform_batch(seed_fn, B, (2 * npair,), device, torch.float32)
    u1 = 1.0 - u[:, :npair]
    u2 = u[:, npair:]
    r = torch.sqrt(-2.0 * torch.log(u1))
    theta = (2.0 * math.pi) * u2
    z = torch.cat([r * torch.cos(theta), r * torch.sin(theta)], dim=1)[:, :n]
    return z.to(dtype).reshape(B, *item_shape)


def device_uniform_batch(seed_fn, B, item_shape, device, dtype=torch.float32):
    """Dispatch on paraug.utils.DEVICE_RNG: 'philox' (default) or 'hash'."""
    from . import utils as _u
    fn = _hash_uniform_batch if getattr(_u, "DEVICE_RNG", "philox") == "hash" else philox_uniform_batch
    return fn(seed_fn, B, item_shape, device, dtype)


def device_normal_batch(seed_fn, B, item_shape, device, dtype=torch.float32):
    from . import utils as _u
    fn = _hash_normal_batch if getattr(_u, "DEVICE_RNG", "philox") == "hash" else philox_normal_batch
    return fn(seed_fn, B, item_shape, device, dtype)


def _philox_tensor_key(c0, c1, c2, c3, k0: torch.Tensor, k1: torch.Tensor):
    """Philox4x32-10 with per-row tensor keys (shape (B,1)) — same math as philox4x32_10."""
    for r in range(10):
        hi0, lo0 = _mulhilo32(_M0, c0)
        hi1, lo1 = _mulhilo32(_M1, c2)
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0), lo1, (hi0 ^ c3 ^ k1), lo0
        if r < 9:
            k0 = (k0 + _W0) & M32
            k1 = (k1 + _W1) & M32
    return c0, c1, c2, c3
