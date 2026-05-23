"""Curated config presets for `AugPipeline`.

Hand-tuned config dictionaries that combine paraug primitives at strengths
that match a particular deployment domain. Use these directly or copy &
adjust:

    from paraug import AugPipeline, presets
    aug = AugPipeline(presets.OOD_PRINTED_PAPER, canvas_size=(512, 512))
    img, mask = aug.compose(fg, bg, mask, seed_base=42)

Each preset is a plain dict — callers can deepcopy and mutate individual
keys to tune for their specific dataset.
"""
from copy import deepcopy


def _copy(d):
    """Return an independent deep copy so callers' mutations don't leak."""
    return deepcopy(d)


# ─── OOD_PRINTED_PAPER ────────────────────────────────────────────────
#
# Tuned for the "printed ECG paper photographed by a phone in indoor
# lighting" deployment: a paper sheet sits on a desk / floor, photo is
# taken at handheld angle, indoor LEDs cast a non-uniform colour, paper
# surface produces a broad specular glare on one side. The target
# segmentation is the ECG content rectangle inside the paper (with white
# margins around it).
#
# Recommended pairing:
#   - call `place_into_canvas(ECG_content, ECG_mask, canvas_size, fill=paper_tone)`
#     first to embed the content in a white-paper canvas with random
#     margins (so the model learns "the wider paper rectangle is a
#     distractor, predict only ECG content");
#   - pass `photo_dir` in `background_compose` pointing to a directory of
#     real desk / floor / scene photos so the model sees realistic
#     backgrounds instead of DTD-only textures.
_OOD_PRINTED_PAPER = {
    "geometric": {
        # Handheld perspective + camera lens warp
        "perspective":        {"p": 0.50, "max_disp_frac": 0.10},
        "optical_distortion": {"p": 0.30, "k": 0.20},
        # Paper-on-surface bending (small TPS) + finger shadow
        "tps":                {"p": 0.30, "max_disp": 12.0, "n_ctrl": 5},
        "random_shadow":      {"p": 0.40, "strength": 0.45, "softness_px": 25.0},
    },
    "photometric": {
        # Background: prefer real photos, fall back to DTD then procedural
        "background_compose": {
            "p": 1.00,
            "p_photo": 0.60,                # high — real photos dominant
            "p_dtd": 0.30,                  # fallback texture
            "bg_brightness_range": (0.5, 1.0),
            "channel_jitter": 0.12,
            # photo_dir / dtd_cache_path are caller-supplied:
            # presets.OOD_PRINTED_PAPER["photometric"]["background_compose"]["photo_dir"] = "..."
        },
        # Lighting realism
        "lighting":           {"p": 0.60, "strength": 0.35, "sigma_frac": 0.45},
        "spatial_color_cast": {"p": 0.50, "amplitude": 0.10, "sigma": 8.0},
        "white_balance_shift": {"p": 0.50, "temp_range_K": (3000.0, 8000.0)},
        # Glare + finger highlight
        "paper_glare":        {"p": 0.40, "area_frac_range": (0.12, 0.35),
                                 "aspect_range": (1.5, 4.0),
                                 "intensity_range": (0.25, 0.55),
                                 "softness": 0.45},
        "specular_streaks":   {"p": 0.30, "n_range": (1, 2),
                                 "length_frac_range": (0.25, 0.55),
                                 "thickness_px": 6.0,
                                 "intensity_range": (0.30, 0.50)},
        # Focus / blur
        "defocus_blur":       {"p": 0.30, "sigma_range": (1.5, 3.0),
                                 "focus_frac_range": (0.20, 0.45),
                                 "n_focus_regions": 1},
        "motion_blur":        {"p": 0.20, "length_range": (3, 9)},
        # Capture-pipeline artefacts
        "jpeg_approx":        {"p": 0.50},   # default q range (lab can tighten)
        "gaussian_noise":     {"p": 0.30, "sigma_range": (0.01, 0.04)},
        # Mild colour / contrast jitter on top
        "color_jitter":       {"p": 0.40, "brightness": 0.20,
                                 "contrast": 0.20, "saturation": 0.15},
        "gamma":              {"p": 0.30, "gamma_range": (0.75, 1.30)},
        "vignette":           {"p": 0.30, "strength_range": (0.25, 0.50)},
        "clahe":              {"p": 0.20},
    },
}


def OOD_PRINTED_PAPER():
    """Independent copy of the printed-paper-OOD preset (see module
    docstring). Pair with `place_into_canvas` for the layout aug."""
    return _copy(_OOD_PRINTED_PAPER)


__all__ = ["OOD_PRINTED_PAPER"]
