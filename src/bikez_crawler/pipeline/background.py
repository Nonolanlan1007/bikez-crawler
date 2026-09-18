"""Background removal for illustrations on a near-uniform white/black background.

Photographs are left alone -- background removal only makes sense for flat catalog
illustrations. Uses BiRefNet (via ``rembg``) rather than a color-distance cutout: telling
a soft ground shadow apart from a translucent windshield, or a shaded swingarm apart from
a shadow's own gradient, needs actual scene understanding, not a color threshold.
"""

from __future__ import annotations

from typing import Any, cast

from PIL import Image
from rembg import new_session, remove

WHITE_THRESHOLD = 235.0
BLACK_THRESHOLD = 20.0
UNIFORMITY_TOLERANCE = 12.0
MIN_AGREEING_FRACTION = 0.8

REMBG_MODEL = "birefnet-general"


def _sample_points(size: tuple[int, int]) -> list[tuple[int, int]]:
    """The four true corners plus edge midpoints, deliberately avoiding the bottom-center
    point where studio product shots often have a soft drop shadow under the bike."""
    width, height = size
    margin_x = max(1, width // 20)
    margin_y = max(1, height // 20)
    return [
        (margin_x, margin_y),
        (width - margin_x - 1, margin_y),
        (margin_x, height - margin_y - 1),
        (width - margin_x - 1, height - margin_y - 1),
        (width // 2, margin_y),
        (margin_x, height // 2),
        (width - margin_x - 1, height // 2),
    ]


def has_uniform_light_or_dark_background(image: Image.Image) -> bool:
    """Cheap check via a handful of corner/edge pixel samples (no model needed).

    Requires only a majority of samples to agree, rather than all of them, so a single
    outlier (a drop shadow, a stray watermark-inpainting artifact) doesn't wrongly veto an
    otherwise-uniform background.
    """
    rgb = image.convert("RGB")
    pixels = [cast("tuple[int, int, int]", rgb.getpixel(point)) for point in _sample_points(rgb.size)]
    grays = sorted(sum(pixel) / 3 for pixel in pixels)
    median = grays[len(grays) // 2]
    agreeing = [gray for gray in grays if abs(gray - median) <= UNIFORMITY_TOLERANCE]
    if len(agreeing) / len(grays) < MIN_AGREEING_FRACTION:
        return False
    average = sum(agreeing) / len(agreeing)
    return average >= WHITE_THRESHOLD or average <= BLACK_THRESHOLD


class BackgroundRemover:
    """Loads the BiRefNet session once per process and reuses it across images."""

    def __init__(self) -> None:
        self._session = new_session(REMBG_MODEL)

    def remove(self, image: Image.Image) -> Image.Image:
        result: Any = remove(image.convert("RGB"), session=self._session)
        if isinstance(result, Image.Image):
            return result
        return Image.fromarray(result)
