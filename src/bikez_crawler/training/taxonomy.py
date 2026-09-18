"""Zero-shot label taxonomy for the ``category`` and ``subject`` classifiers.

This is a *starter* taxonomy meant to be revised once the review tool (see
``scripts/generate_review_page.py``) surfaces how well it actually matches the gallery
photos. To revise it: add/remove/reword prompts below and re-run
``scripts/bootstrap_labels.py`` — nothing else needs to change, the training and review
tooling both read the label sets from here.

Each label maps to a handful of prompt variants; CLIP scores an image against the mean
of each label's prompt embeddings (prompt ensembling), which is more robust than a
single template per class.

``category`` (view/framing of the shot):
  - front / rear / side / front_three_quarter / rear_three_quarter / top: whole-bike
    shots from a given angle, no rider. "three_quarter" was renamed to
    "front_three_quarter" when "rear_three_quarter" was added, since the original
    prompts always meant the front angle -- keeping a bare "three_quarter" alongside
    "rear_three_quarter" would have been ambiguous.
  - riding: someone actively riding the bike (any angle) -- takes precedence over the
    angle labels above, since "there's a rider on it" is the salient fact about these
    shots, not which side the camera happens to be on.
  - engine_detail / wheel_detail / brake_lever_detail / dashboard_cockpit /
    exhaust_detail / seat_detail / fairing_detail: close-up "zoom" shots of one part,
    which show up inconsistently across bikes' galleries (some bikes have none, some
    have several).
  - other_detail: catch-all for close-ups that don't fit a named part, so the
    classifier isn't forced to mislabel them into one of the above.

``subject`` (studio vs. in-the-world):
  - illustration: clean manufacturer photo/render, plain background, no rider, no scenery.
  - photograph: a real-world shot, e.g. someone riding it on a road, in a desert, etc.
"""

from __future__ import annotations

CATEGORY_PROMPTS: dict[str, list[str]] = {
    "front": [
        "a front view photo of a motorcycle",
        "a motorcycle photographed head-on from the front",
        "a motorcycle's front fairing and headlight, shot straight on",
    ],
    "rear": [
        "a rear view photo of a motorcycle",
        "a motorcycle photographed from directly behind",
        "a motorcycle's taillight and rear end, shot straight on",
    ],
    "side": [
        "a side view photo of a motorcycle",
        "a motorcycle profile shot from the side",
        "a motorcycle photographed from directly to the side, full profile",
    ],
    "front_three_quarter": [
        "a three-quarter angle photo of a motorcycle",
        "a motorcycle shot from a three-quarter front angle",
        "an angled photo of a motorcycle showing both the front and the side",
    ],
    "rear_three_quarter": [
        "a three-quarter angle photo of a motorcycle from the rear",
        "a motorcycle shot from a three-quarter rear angle",
        "an angled photo of a motorcycle showing both the rear and the side",
    ],
    "top": [
        "a top-down view photo of a motorcycle",
        "a motorcycle photographed from directly above",
        "an overhead, bird's-eye view shot of a motorcycle",
    ],
    "riding": [
        "a photo of a person riding a motorcycle",
        "someone riding a motorcycle on a road or trail",
        "an action photo of a rider on a motorcycle in motion",
    ],
    "engine_detail": [
        "a close-up photo of a motorcycle engine",
        "a zoomed-in detail shot of a motorcycle engine block and cylinders",
    ],
    "wheel_detail": [
        "a close-up photo of a motorcycle wheel and tire",
        "a zoomed-in detail shot of a motorcycle rim, spokes, and tire",
    ],
    "brake_lever_detail": [
        "a close-up photo of a motorcycle brake lever",
        "a zoomed-in detail shot of a motorcycle handlebar and brake lever",
    ],
    "dashboard_cockpit": [
        "a close-up photo of a motorcycle dashboard and instrument cluster",
        "a photo of a motorcycle cockpit view showing the speedometer and gauges",
    ],
    "exhaust_detail": [
        "a close-up photo of a motorcycle exhaust pipe and muffler",
    ],
    "fairing_detail": [
        "a close-up photo of a motorcycle fairing panel",
        "a zoomed-in detail shot of a motorcycle's plastic bodywork and paintwork",
        "a close-up of a motorcycle cowling showing its shape and decals",
    ],
    "seat_detail": [
        "a close-up photo of a motorcycle seat",
    ],
    "other_detail": [
        "a close-up detail photo of an unspecified motorcycle part",
        "an abstract close-up shot of a motorcycle component",
    ],
}

SUBJECT_PROMPTS: dict[str, list[str]] = {
    "illustration": [
        "a studio photo of a motorcycle on a plain background",
        "a manufacturer product shot of a motorcycle with no rider and no scenery",
        "a clean studio render of a motorcycle isolated on a plain backdrop",
    ],
    "photograph": [
        "a photograph of a person riding a motorcycle outdoors",
        "a motorcycle being ridden on a road or trail in a real landscape",
        "an action photo of a motorcycle in a real-world environment, such as a desert or city street",
    ],
}
