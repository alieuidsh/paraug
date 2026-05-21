"""04 — compositing and layered synthesis with compose().

`compose(foreground, background, mask)` blends a foreground onto a
background through a mask, then runs the configured aug. Calling it twice
builds a layered scene: pass-1 output becomes pass-2's foreground.

Before running, install paraug (e.g. `pip install -e .` from repo root),
then:
    python examples/04_compose_layered.py
"""
import numpy as np

from paraug import AugPipeline


def main():
    H, W = 96, 128

    # Pass 1 — "content printed on paper": ECG-like content onto a paper tone.
    content = np.full((H, W, 3), 60, dtype=np.uint8)        # dark content
    paper_tone = np.full((H, W, 3), 245, dtype=np.uint8)    # near-white paper
    content_mask = np.zeros((H, W), dtype=np.uint8)
    content_mask[H // 4: 3 * H // 4, W // 4: 3 * W // 4] = 255  # inner region

    # Pass 2 — "paper photographed in a scene": the printed sheet onto a scene.
    scene_bg = np.full((H, W, 3), 30, dtype=np.uint8)       # dark scene
    paper_outline = np.full((H, W), 255, dtype=np.uint8)    # whole sheet shows

    # canvas_size conforms every output to a fixed size for uniform batching.
    aug = AugPipeline(
        {
            "geometric":   {"affine": {"p": 1.0, "rot_deg": 8.0,
                                        "scale_range": (0.9, 1.1)}},
            "photometric": {"gamma": {"p": 0.5, "gamma_range": (0.8, 1.2)}},
        },
        canvas_size=(256, 256),
    )

    img1, mask1, t1 = aug.compose(content, paper_tone, content_mask,
                                   seed_base=0, return_transform=True)
    print(f"pass 1 (printing):      img {img1.shape}  mask {mask1.shape}")
    print(f"  canvas stretch {content.shape[:2]} → {tuple(t1['canvas_size'])}: "
          f"scale_x={t1['scale_x']:.3f} scale_y={t1['scale_y']:.3f}")

    # Pass-1 output feeds pass-2 foreground. NB: pass 2's mask is the PAPER
    # SHEET outline, not the pass-1 content mask — otherwise the paper border
    # would be replaced by the scene. scene_bg / paper_outline are authored at
    # the original size; compose resizes them to the (canvas-sized) pass-1
    # foreground automatically.
    img2, mask2 = aug.compose(img1, scene_bg, paper_outline, seed_base=1)
    print(f"pass 2 (photographing): img {img2.shape}  mask {mask2.shape}")

    assert img2.shape == (256, 256, 3), "canvas_size conformed the output"
    assert img2.dtype == np.uint8, "numpy uint8 in → numpy uint8 out"


if __name__ == "__main__":
    main()
