"""Tests for AugPipeline.compose() and canvas_size (paraug v0.4.0).

compose() blends a foreground onto a background through a mask then runs
the configured aug. canvas_size conforms every output to a fixed size via
non-uniform stretch.
"""
import numpy as np
import pytest
import torch

from paraug import AugPipeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 64, 80


# ─── compose: blend correctness ───────────────────────────────────────
def test_compose_blend_numpy_no_aug():
    """Empty config → compose is a pure blend. numpy HWC uint8 in/out."""
    fg = np.zeros((H, W, 3), dtype=np.uint8); fg[..., 0] = 255   # red
    bg = np.zeros((H, W, 3), dtype=np.uint8); bg[..., 2] = 255   # blue
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[:, : W // 2] = 255                                       # left half fg

    aug = AugPipeline({})  # no primitives
    img, mask_out = aug.compose(fg, bg, mask, seed_base=0)

    assert isinstance(img, np.ndarray), "numpy in → numpy out"
    assert img.dtype == np.uint8
    assert img.shape == (H, W, 3)
    # Left half: red foreground
    assert (img[:, : W // 2, 0] > 250).all(), "left half should be red fg"
    assert (img[:, : W // 2, 2] < 5).all()
    # Right half: blue background
    assert (img[:, W // 2:, 2] > 250).all(), "right half should be blue bg"
    assert (img[:, W // 2:, 0] < 5).all()
    assert mask_out.shape == (H, W)


def test_compose_blend_tensor_no_aug():
    """Tensor in → tensor out, same blend semantics."""
    fg = torch.zeros(1, 3, H, W, device=DEVICE); fg[:, 0] = 1.0
    bg = torch.zeros(1, 3, H, W, device=DEVICE); bg[:, 2] = 1.0
    mask = torch.zeros(1, 1, H, W, device=DEVICE)
    mask[:, :, :, : W // 2] = 1.0

    aug = AugPipeline({})
    img, mask_out = aug.compose(fg, bg, mask, seed_base=0)

    assert torch.is_tensor(img), "tensor in → tensor out"
    assert img.shape == (1, 3, H, W)
    assert (img[:, 0, :, : W // 2] > 0.99).all()
    assert (img[:, 2, :, W // 2:] > 0.99).all()
    assert mask_out.shape == (1, 1, H, W)


# ─── compose: aug applied ─────────────────────────────────────────────
def test_compose_photometric_applies_to_composite():
    """A photometric primitive should change the composite."""
    fg = torch.full((1, 3, H, W), 0.5, device=DEVICE)
    bg = torch.full((1, 3, H, W), 0.5, device=DEVICE)
    mask = torch.ones(1, 1, H, W, device=DEVICE)  # all foreground
    aug = AugPipeline({"photometric": {"gamma": {"p": 1.0,
                                                   "gamma_range": (0.5, 0.5)}}})
    img, _ = aug.compose(fg, bg, mask, seed_base=1)
    # gamma 0.5 on 0.5 → 0.5^0.5 ≈ 0.707
    assert (img - 0.5).abs().max() > 0.1, "gamma should have changed composite"


def test_compose_geometric_warps_fg_and_mask_together():
    """Geometric warp must move foreground and mask in lockstep — the blend
    boundary stays consistent with the warped mask."""
    fg = torch.zeros(1, 3, H, W, device=DEVICE); fg[:, 0] = 1.0
    bg = torch.zeros(1, 3, H, W, device=DEVICE); bg[:, 2] = 1.0
    mask = torch.zeros(1, 1, H, W, device=DEVICE)
    mask[:, :, H // 4: 3 * H // 4, W // 4: 3 * W // 4] = 1.0  # centre block
    aug = AugPipeline({"geometric": {"affine": {"p": 1.0, "rot_deg": 20.0}}})
    img, mask_out = aug.compose(fg, bg, mask, seed_base=2)
    # Wherever the warped mask is ~1, the composite must be foreground (red);
    # wherever ~0, background (blue). Check the blend honoured the warped mask.
    m = mask_out[0, 0]
    fg_region = m > 0.99
    bg_region = m < 0.01
    if fg_region.any():
        assert img[0, 0][fg_region].mean() > 0.9, "fg region should be red"
    if bg_region.any():
        assert img[0, 2][bg_region].mean() > 0.9, "bg region should be blue"


# ─── canvas_size ──────────────────────────────────────────────────────
def test_canvas_size_call():
    """__call__ output is stretched to canvas_size."""
    img = torch.rand(2, 3, 100, 120, device=DEVICE)
    mask = torch.ones(2, 1, 100, 120, device=DEVICE)
    aug = AugPipeline({"geometric": {"affine": {"p": 1.0, "rot_deg": 10.0}}},
                      canvas_size=(64, 48))
    out, mask_out = aug(img, mask=mask, seed_base=0)
    assert out.shape == (2, 3, 64, 48), f"got {tuple(out.shape)}"
    assert mask_out.shape == (2, 1, 64, 48)


def test_canvas_size_compose():
    """compose output is stretched to canvas_size."""
    fg = np.zeros((100, 120, 3), dtype=np.uint8)
    bg = np.zeros((100, 120, 3), dtype=np.uint8)
    mask = np.zeros((100, 120), dtype=np.uint8)
    aug = AugPipeline({}, canvas_size=(64, 48))
    img, mask_out = aug.compose(fg, bg, mask, seed_base=0)
    assert img.shape == (64, 48, 3), f"got {img.shape}"
    assert mask_out.shape == (64, 48)


def test_canvas_size_none_keeps_input_size():
    """canvas_size None (default) leaves output size equal to input."""
    img = torch.rand(1, 3, 77, 91, device=DEVICE)
    aug = AugPipeline({"geometric": {"affine": {"p": 1.0}}})
    out, _ = aug(img, seed_base=0)
    assert out.shape == (1, 3, 77, 91)


def test_canvas_size_transform_dict():
    """return_transform reports the canvas stretch factors."""
    fg = np.zeros((100, 200, 3), dtype=np.uint8)
    bg = np.zeros((100, 200, 3), dtype=np.uint8)
    mask = np.zeros((100, 200), dtype=np.uint8)
    aug = AugPipeline({}, canvas_size=(50, 400))
    img, mask_out, transform = aug.compose(fg, bg, mask, seed_base=0,
                                            return_transform=True)
    assert transform["canvas_size"] == (50, 400)
    assert abs(transform["scale_x"] - 400 / 200) < 1e-9, "scale_x = canvas_W/in_W"
    assert abs(transform["scale_y"] - 50 / 100) < 1e-9, "scale_y = canvas_H/in_H"


def test_compose_transform_dict_no_canvas():
    """Without canvas_size the transform scales are 1.0."""
    fg = np.zeros((H, W, 3), dtype=np.uint8)
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)
    aug = AugPipeline({})
    _, _, transform = aug.compose(fg, bg, mask, return_transform=True)
    assert transform["scale_x"] == 1.0 and transform["scale_y"] == 1.0
    assert transform["canvas_size"] is None


# ─── validation ───────────────────────────────────────────────────────
def test_compose_size_mismatch_resizes_to_foreground():
    """background / mask are resized to the foreground frame, not an error —
    so a layered compose chain works without pre-sizing each input."""
    fg = np.zeros((H, W, 3), dtype=np.uint8); fg[..., 1] = 255   # green fg
    bg = np.zeros((H * 2, W + 30, 3), dtype=np.uint8)            # mismatched bg
    bg[..., 2] = 255                                             # blue bg
    mask = np.full((H // 2, W - 10), 255, dtype=np.uint8)        # mismatched mask
    aug = AugPipeline({})
    img, mask_out = aug.compose(fg, bg, mask)
    assert img.shape == (H, W, 3), "output conforms to foreground frame"
    assert mask_out.shape == (H, W)
    # mask resized to all-255 → composite is the green foreground everywhere
    assert (img[..., 1] > 250).all()


def test_compose_channel_mismatch_raises():
    fg = np.zeros((H, W, 3), dtype=np.uint8)
    bg = np.zeros((H, W, 1), dtype=np.uint8)   # 1-channel bg vs 3-channel fg
    mask = np.zeros((H, W), dtype=np.uint8)
    aug = AugPipeline({})
    with pytest.raises(ValueError, match="channels"):
        aug.compose(fg, bg, mask)


def test_compose_layered_chain_with_canvas():
    """Layered compose under canvas_size: pass-1 output (canvas-sized) feeds
    pass-2 foreground; pass-2 bg/mask (original size) auto-resize."""
    aug = AugPipeline({}, canvas_size=(200, 240))
    content = np.full((H, W, 3), 60, dtype=np.uint8)
    paper = np.full((H, W, 3), 240, dtype=np.uint8)
    cmask = np.zeros((H, W), dtype=np.uint8); cmask[10:-10, 10:-10] = 255
    scene = np.full((H, W, 3), 20, dtype=np.uint8)
    sheet = np.full((H, W), 255, dtype=np.uint8)
    img1, m1 = aug.compose(content, paper, cmask)          # → (200, 240, 3)
    assert img1.shape == (200, 240, 3)
    img2, m2 = aug.compose(img1, scene, sheet)             # bg/mask resize up
    assert img2.shape == (200, 240, 3)


def test_canvas_size_invalid_raises():
    with pytest.raises(ValueError, match="canvas_size must be"):
        AugPipeline({}, canvas_size=(0, 100))
    with pytest.raises(ValueError, match="canvas_size must be"):
        AugPipeline({}, canvas_size=(64,))


def test_compose_layered_two_calls():
    """Double-layer synthesis: pass-1 output feeds pass-2 foreground."""
    ecg = np.full((H, W, 3), 200, dtype=np.uint8)
    paper = np.full((H, W, 3), 250, dtype=np.uint8)
    ecg_mask = np.zeros((H, W), dtype=np.uint8)
    ecg_mask[H // 4: 3 * H // 4, W // 4: 3 * W // 4] = 255
    scene = np.full((H, W, 3), 30, dtype=np.uint8)
    paper_mask = np.full((H, W), 255, dtype=np.uint8)

    aug = AugPipeline({})
    img1, _ = aug.compose(ecg, paper, ecg_mask)
    img2, mask2 = aug.compose(img1, scene, paper_mask)
    assert img2.shape == (H, W, 3)
    # paper_mask all-foreground → img2 == img1 (scene fully covered)
    assert np.array_equal(img2, img1)


# ─── CPU/GPU parity ───────────────────────────────────────────────────
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compose_cpu_cuda_parity():
    """compose is bit-exact CPU vs CUDA within grid_sample tolerance."""
    g = torch.Generator().manual_seed(0)
    fg = torch.rand(2, 3, H, W, generator=g)
    bg = torch.rand(2, 3, H, W, generator=g)
    mask = (torch.rand(2, 1, H, W, generator=g) > 0.5).float()
    cfg = {"geometric": {"affine": {"p": 1.0, "rot_deg": 12.0}},
           "photometric": {"gamma": {"p": 1.0, "gamma_range": (0.8, 1.2)}}}
    aug = AugPipeline(cfg, canvas_size=(48, 56))
    img_cpu, m_cpu = aug.compose(fg, bg, mask, seed_base=7)
    img_cu, m_cu = aug.compose(fg.cuda(), bg.cuda(), mask.cuda(), seed_base=7)
    assert (img_cpu - img_cu.cpu()).abs().max() < 2e-4
    assert (m_cpu - m_cu.cpu()).abs().max() < 2e-4


# ─── v0.5.2: aug does not propagate gradients (VRAM regression test) ──
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compose_no_grad_on_graph_inputs():
    """Aug is preprocessing — even if inputs carry an autograd graph (e.g. the
    caller forgot to .detach() after a TPS warp), aug must not pile intermediates
    on it. Reproducing the v0.5.0 bug: tensors with requires_grad=True passed
    into compose() previously kept every primitive's intermediate alive, adding
    ~5 GB at bs=20 canvas=1024. Fix: __call__/compose wrap body in no_grad."""
    aug = AugPipeline({"geometric": {"affine": {"p": 1.0, "rot_deg": 5.0}},
                       "photometric": {"gamma": {"p": 1.0, "gamma_range": (0.8, 1.2)}}})
    fg = torch.rand(2, 3, 32, 40, device="cuda", requires_grad=True)
    bg = torch.rand(2, 3, 32, 40, device="cuda", requires_grad=True)
    mask = (torch.rand(2, 1, 32, 40, device="cuda") > 0.5).float()
    img, m = aug.compose(fg, bg, mask, seed_base=0)
    # Outputs must be detached from the input graph.
    assert not img.requires_grad, "compose output still attached to autograd graph"
    if m is not None:
        assert not m.requires_grad, "compose mask output still attached to autograd graph"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_call_no_grad_on_graph_inputs():
    """Same as test_compose_no_grad_on_graph_inputs, but for __call__."""
    aug = AugPipeline({"geometric": {"affine": {"p": 1.0, "rot_deg": 5.0}},
                       "photometric": {"gamma": {"p": 1.0, "gamma_range": (0.8, 1.2)}}})
    img_in = torch.rand(2, 3, 32, 40, device="cuda", requires_grad=True)
    mask_in = (torch.rand(2, 1, 32, 40, device="cuda") > 0.5).float()
    img, m = aug(img_in, mask=mask_in, seed_base=0)
    assert not img.requires_grad, "__call__ output still attached to autograd graph"
    if m is not None:
        assert not m.requires_grad, "__call__ mask output still attached to autograd graph"


# ─── v0.5.4: chunk_size VRAM bound (output != unchunked by design) ────
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_chunk_size_correctness_deterministic_per_seed():
    """chunk_size produces deterministic output per (seed_base, chunk_size).
    Same seed_base + chunk_size → identical output across calls."""
    cfg = {"geometric": {"perspective": {"p": 1.0, "scale_range": (0.1, 0.2)}},
           "photometric": {"gamma": {"p": 1.0, "gamma_range": (0.7, 1.3)}}}
    fg = torch.rand(8, 3, 32, 32, device="cuda")
    bg = torch.rand(8, 3, 32, 32, device="cuda")
    mask = (torch.rand(8, 1, 32, 32, device="cuda") > 0.5).float()
    aug = AugPipeline(cfg, chunk_size=4)
    img1, m1 = aug.compose(fg, bg, mask, seed_base=42)
    img2, m2 = aug.compose(fg, bg, mask, seed_base=42)
    assert torch.equal(img1, img2)
    assert torch.equal(m1, m2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_chunk_size_distinct_items_across_chunks():
    """Items at local position 0 in chunk 0 vs chunk 1 must get distinct
    augmentation — otherwise items 0 and chunk_size would collide."""
    cfg = {"photometric": {"gamma": {"p": 1.0, "gamma_range": (0.5, 1.5)}}}
    fg = torch.ones(4, 3, 16, 16, device="cuda") * 0.5
    bg = torch.zeros(4, 3, 16, 16, device="cuda")
    mask = torch.ones(4, 1, 16, 16, device="cuda")
    aug = AugPipeline(cfg, chunk_size=2)  # 4 items / chunk 2 = 2 chunks
    img, _ = aug.compose(fg, bg, mask, seed_base=7)
    # Item 0 and item 2 are local position 0 in their respective chunks.
    # With the chunk-seed offset they should differ (different gamma applied).
    assert not torch.equal(img[0], img[2]), "chunk seed offset failed — items collided"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_chunk_size_unchunked_matches_when_batch_fits():
    """When B <= chunk_size, chunked path is bypassed → output identical
    to the no-chunk path."""
    cfg = {"photometric": {"gamma": {"p": 1.0, "gamma_range": (0.8, 1.2)}}}
    fg = torch.rand(4, 3, 16, 16, device="cuda")
    bg = torch.rand(4, 3, 16, 16, device="cuda")
    mask = (torch.rand(4, 1, 16, 16, device="cuda") > 0.5).float()
    a1 = AugPipeline(cfg, chunk_size=None)
    a2 = AugPipeline(cfg, chunk_size=8)  # B=4 <= 8, no chunking
    img1, _ = a1.compose(fg, bg, mask, seed_base=11)
    img2, _ = a2.compose(fg, bg, mask, seed_base=11)
    assert torch.equal(img1, img2)


# ─── v0.6.0: small packaging niceties ─────────────────────────────────
def test_list_primitives_geometric():
    from paraug.geometric import list_primitives, GEOMETRIC_PRIMITIVES
    lp = list_primitives()
    assert isinstance(lp, list)
    assert lp == sorted(lp)  # deterministic order
    assert set(lp) == set(GEOMETRIC_PRIMITIVES.keys())


def test_list_primitives_photometric():
    from paraug.photometric import list_primitives, PHOTOMETRIC_PRIMITIVES
    lp = list_primitives()
    assert isinstance(lp, list)
    assert lp == sorted(lp)
    assert set(lp) == set(PHOTOMETRIC_PRIMITIVES.keys())


def test_pipeline_accepts_config_none():
    """AugPipeline(config=None) and AugPipeline() should both be valid (empty pipeline)."""
    a1 = AugPipeline()
    a2 = AugPipeline(None)
    a3 = AugPipeline({})
    # All three behave as a no-op pipeline.
    img = torch.rand(2, 3, 16, 16)
    mask = (torch.rand(2, 1, 16, 16) > 0.5).float()
    out1, _ = a1(img, mask, seed_base=0)
    out2, _ = a2(img, mask, seed_base=0)
    out3, _ = a3(img, mask, seed_base=0)
    assert torch.equal(out1, out2) and torch.equal(out2, out3)


def test_submodule_reexport():
    import paraug
    # Public submodule access for callers that want list_primitives().
    assert hasattr(paraug, "geometric")
    assert hasattr(paraug, "photometric")
    assert callable(paraug.geometric.list_primitives)
    assert callable(paraug.photometric.list_primitives)
