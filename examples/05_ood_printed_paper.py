"""05 — OOD printed-paper preset, end-to-end.

Build a synthetic "printed ECG paper photographed by a phone on a desk"
sample using v0.5.0 features:
  1. `place_into_canvas` to embed the ECG content rectangle inside a
     paper-sheet canvas at a random off-centre position with white margins
  2. `AugPipeline(presets.OOD_PRINTED_PAPER(), canvas_size=...)` to apply
     the photo-realism aug (background, glare, WB, defocus, etc.)

Before running, install paraug (`pip install -e .` from repo root), then:
    python examples/05_ood_printed_paper.py
"""
import numpy as np

from paraug import AugPipeline, place_into_canvas, presets


def main():
    # ECG content: a pink-grid block (small rectangle vs. the paper canvas).
    content_h, content_w = 240, 360
    ecg = np.full((content_h, content_w, 3), 245, dtype=np.uint8)
    # Mock pink grid lines every 12 px
    ecg[::12, :, 0] = 230; ecg[::12, :, 1] = 200; ecg[::12, :, 2] = 210
    ecg[:, ::12, 0] = 230; ecg[:, ::12, 1] = 200; ecg[:, ::12, 2] = 210
    # Mock dark ECG trace
    ecg[content_h // 2 - 2: content_h // 2 + 2, :, :] = 40

    ecg_mask = np.full((content_h, content_w), 255, dtype=np.uint8)

    # Step 1 — embed ECG content into a paper-sheet canvas with random margins.
    paper_h, paper_w = 600, 800
    paper_tone = (248, 248, 246)   # off-white paper
    paper_padded, mask_padded = place_into_canvas(
        ecg, ecg_mask,
        canvas_size=(paper_h, paper_w),
        fill=paper_tone,
        margin_frac_range=(0.05, 0.30),
        seed_base=42, epoch=0, step=0,
    )
    print(f"Paper canvas: {paper_padded.shape}  mask: {mask_padded.shape}")
    print(f"  mask region: {int(mask_padded.sum() // 255)} px ({mask_padded.mean()/255:.1%} "
          f"of paper)")

    # Step 2 — compose onto a scene background + apply OOD photo-realism aug.
    # NB: in production replace `scene_bg` with real desk / floor photos via
    # `presets.OOD_PRINTED_PAPER()["photometric"]["background_compose"]
    #     ["photo_dir"] = "/path/to/scene_photos"`.
    scene_bg = np.full((paper_h, paper_w, 3), 90, dtype=np.uint8)
    paper_outline = np.full((paper_h, paper_w), 255, dtype=np.uint8)

    cfg = presets.OOD_PRINTED_PAPER()
    cfg["photometric"].pop("background_compose", None)   # no photo dir for demo
    aug = AugPipeline(cfg, canvas_size=(512, 640))

    img, mask = aug.compose(paper_padded, scene_bg, paper_outline,
                             seed_base=42)
    print(f"Final synthetic sample: img {img.shape}  mask {mask.shape}")
    print(f"  final mask (paper region) area: {int(mask.sum() // 255)} px")

    # The training loop would now use `img` as model input and
    # `place_into_canvas`'s `mask_padded` (resized to the final canvas) as
    # the loss target — the ECG-content region inside the wider paper.
    # That nested-rectangle layout is what teaches the model to skip the
    # outer paper boundary at inference time.


if __name__ == "__main__":
    main()
