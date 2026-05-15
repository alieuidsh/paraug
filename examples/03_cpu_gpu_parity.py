"""03 — CPU/GPU bit-exact parity demonstration.

Same `(seed_base, epoch, step)` produces the same output on CPU and CUDA
within 2e-4 (grid_sample-class tolerance) for geometric ops, and 1e-6 for
pure elementwise photometric ops. paraug's USP — most aug libraries don't
guarantee this.

Before running, install paraug (e.g. `pip install -e .` from repo root or
`PYTHONPATH=src python ...`), then:
    python examples/03_cpu_gpu_parity.py
"""
import torch

from paraug import AugPipeline


def main():
    if not torch.cuda.is_available():
        print("=" * 72)
        print("⚠️  CUDA NOT AVAILABLE — CUDA parity demo SKIPPED.")
        print("=" * 72)
        print("This example only exercises paraug on CPU; the headline 1e-6 /")
        print("2e-4 CPU↔CUDA tolerance claim is NOT verified by this run.")
        print()
        print("To verify locally on a CUDA-enabled machine:")
        print("    pip install -e .")
        print("    python examples/03_cpu_gpu_parity.py")
        print("    pytest tests/test_parity.py -k cpu_vs_cuda  # 13 tests")
        print()
        print("See README parity-comparison section for the verified hardware")
        print("at v0.1.0 (NVIDIA 5060 Ti / 4080).")
        print("=" * 72)
        return

    B, H, W = 2, 128, 128
    g = torch.Generator(device="cpu").manual_seed(0)
    img = torch.rand(B, 3, H, W, generator=g)
    mask = torch.ones(B, 1, H, W)

    aug = AugPipeline({
        "geometric": {
            "affine":            {"p": 1.0, "rot_deg": 12.0,
                                   "scale_range": (0.9, 1.1)},
            "tps":               {"p": 1.0, "max_disp": 8.0, "n_ctrl": 5},
            "elastic_transform": {"p": 1.0, "alpha": 6.0},
        },
        "photometric": {
            "gamma":             {"p": 1.0, "gamma_range": (0.8, 1.2)},
            "color_jitter":      {"p": 1.0, "brightness": 0.2,
                                   "contrast": 0.2, "saturation": 0.2},
        },
    })

    out_cpu = aug(img, mask=mask, seed_base=42, epoch=0, step=0)
    out_cuda = aug(img.cuda(), mask=mask.cuda(),
                    seed_base=42, epoch=0, step=0)

    img_diff = (out_cpu[0] - out_cuda[0].cpu()).abs().max().item()
    mask_diff = (out_cpu[1] - out_cuda[1].cpu()).abs().max().item()
    print(f"CPU vs CUDA image max_abs_diff: {img_diff:.2e}")
    print(f"CPU vs CUDA mask  max_abs_diff: {mask_diff:.2e}")
    print(f"tolerance for grid_sample-class ops: 2e-4")
    assert img_diff < 2e-4, f"image parity broken: {img_diff:.2e} > 2e-4"
    assert mask_diff < 2e-4, f"mask parity broken: {mask_diff:.2e} > 2e-4"
    print("PARITY OK")


if __name__ == "__main__":
    main()
