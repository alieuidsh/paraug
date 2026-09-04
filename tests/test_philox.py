"""Device-side Philox RNG: known-answer vectors, cross-device bit-exactness,
per-item batch independence, and CPU/CUDA parity of the noise primitives when
`PARAUG_PHILOX` mode is on."""
import pytest
import torch

from paraug import philox as px
from paraug import utils as u
from paraug import AugPipeline

cuda_available = torch.cuda.is_available()


def _t(v):
    return torch.tensor([v], dtype=torch.int64)


# Random123 philox4x32-10 known-answer vectors (kat_vectors).
KAT = [
    ((0, 0, 0, 0), (0, 0),
     (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2,
     (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
]


@pytest.mark.parametrize("ctr,key,expected", KAT)
def test_philox_known_answer(ctr, key, expected):
    out = px.philox4x32_10(*[_t(c) for c in ctr], *key)
    assert tuple(int(o) for o in out) == expected


def test_mulhilo_matches_python_bigint():
    a = 0xD2511F53
    b = torch.tensor([0, 1, 0xFFFFFFFF, 0x12345678, 0xCD9E8D57], dtype=torch.int64)
    hi, lo = px._mulhilo32(a, b)
    for bi, h, l in zip(b.tolist(), hi.tolist(), lo.tolist()):
        prod = a * bi
        assert h == (prod >> 32) & 0xFFFFFFFF and l == prod & 0xFFFFFFFF


def test_uniform_range_and_stats():
    u_ = px.philox_uniform(42, (512, 512), "cpu")
    assert float(u_.min()) >= 0.0 and float(u_.max()) < 1.0
    assert abs(float(u_.mean()) - 0.5) < 5e-3
    assert abs(float(u_.std()) - 0.2887) < 5e-3


def test_normal_stats_finite():
    z = px.philox_normal(7, (512, 512), "cpu")
    assert torch.isfinite(z).all()
    assert abs(float(z.mean())) < 1e-2 and abs(float(z.std()) - 1.0) < 1e-2


def test_batch_item_independent_of_batch_size():
    seed = lambda i: 1000 + i
    a = px.philox_normal_batch(seed, 2, (3, 16, 16), "cpu")
    b = px.philox_normal_batch(seed, 5, (3, 16, 16), "cpu")
    assert torch.equal(a, b[:2])
    a = px._hash_uniform_batch(seed, 2, (3, 16, 16), "cpu")
    b = px._hash_uniform_batch(seed, 5, (3, 16, 16), "cpu")
    assert torch.equal(a, b[:2])


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_uint32_stream_bit_exact_cpu_vs_cuda():
    a = px.philox_uint32(42, 100_000, "cpu")
    b = px.philox_uint32(42, 100_000, "cuda")
    assert torch.equal(a, b.cpu())
    seed = lambda i: 99 + i
    a = px._hash_uniform_batch(seed, 3, (64, 64), "cpu")
    b = px._hash_uniform_batch(seed, 3, (64, 64), "cuda")
    assert torch.equal(a, b.cpu())


NOISE_SPECS = {
    "gaussian_noise":    {"p": 1.0, "sigma_range": (0.005, 0.04)},
    "jpeg_approx":       {"p": 1.0, "noise_sigma_range": (0.005, 0.02)},
    "salt_pepper_noise": {"p": 1.0, "density": 0.02},
}


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
@pytest.mark.parametrize("name", sorted(NOISE_SPECS))
@pytest.mark.parametrize("backend", ["philox", "hash"])
def test_noise_parity_device_rng(name, backend, monkeypatch):
    monkeypatch.setattr(u, "USE_PHILOX", True)
    monkeypatch.setattr(u, "DEVICE_RNG", backend)
    g = torch.Generator(device="cpu").manual_seed(0)
    img = torch.rand(2, 3, 64, 64, generator=g)
    mask = torch.ones(2, 1, 64, 64)
    aug = AugPipeline({"photometric": {name: NOISE_SPECS[name]}})
    oc, _ = aug(img, mask=mask, seed_base=7, epoch=1, step=3)
    og, _ = aug(img.cuda(), mask=mask.cuda(), seed_base=7, epoch=1, step=3)
    tol = 2e-4 if name == "jpeg_approx" else 1e-6   # jpeg_approx goes through conv2d
    assert (oc - og.cpu()).abs().max().item() <= tol


def test_default_mode_unchanged(monkeypatch):
    """With USE_PHILOX off, the CPU-generator stream is the historical one."""
    monkeypatch.setattr(u, "USE_PHILOX", False)
    g = torch.Generator(device="cpu").manual_seed(0)
    img = torch.rand(1, 3, 32, 32, generator=g)
    aug = AugPipeline({"photometric": {"gaussian_noise": {"p": 1.0, "sigma_range": (0.02, 0.02)}}})
    out, _ = aug(img, seed_base=1, epoch=0, step=0)
    ref_g = u.cpu_generator(u.per_item_seed(1, 0, 0, 0, "gaussian_noise"))
    u.sample_bool(1.0, ref_g); u.sample_uniform(0.02, 0.02, ref_g)
    noise = torch.empty(3, 32, 32).normal_(0.0, 1.0, generator=ref_g) * 0.02
    assert torch.allclose(out[0], (img[0] + noise).clamp(0, 1))
