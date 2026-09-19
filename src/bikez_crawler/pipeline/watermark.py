"""bikez.com watermark removal.

Two watermarks are stamped on the poster images. The large translucent "Bikez.com"
overlay is undone exactly by inverting its alpha blend, so the pixels underneath are
recovered rather than repainted. The opaque bottom-right logo box can't be inverted and
is inpainted instead.

Inpainting uses LaMa (``simple_lama_inpainting``), which reconstructs textured
backgrounds (roads, foliage, studio gradients) far better than classical inpainting.
``cv2.inpaint`` (Telea) is the fallback when LaMa can't be loaded or fails on a given
image, since it's ~200x faster but leaves a visible smear on non-flat backgrounds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# The bottom-right "Bikez / Online Motorcycle Catalog / www.bikez.com" logo box is a
# FIXED PIXEL SIZE graphic, not proportional to the source image's dimensions -- confirmed
# by comparing the identical logo box across images at 1280x853, 1280x905, and 918x629.
# This only holds for the large "poster" source (`_poster.php`, filename variant `_4_`),
# which is what the crawler now fetches instead of the small `_picture.php` thumbnail
# (whose watermark WAS roughly proportional to its ~400x266 size). Sized with a small
# margin beyond the measured logo bounds.
WATERMARK_BOX_WIDTH = 340
WATERMARK_BOX_HEIGHT = 125

CENTER_MASK_PATH = Path(__file__).resolve().parent / "assets" / "center_watermark_mask.png"

# The overlay is a fixed-size stamp anchored to the bottom-right: the mask's top-left
# corner sits at (width - X, height - Y), so it is simply cropped on small images.
_CENTER_MASK_ANCHOR_X = 820
_CENTER_MASK_ANCHOR_Y = 562

# sRGB blend: stamped = (1 - ALPHA * mask) * original + ALPHA * mask * COLOR
_CENTER_ALPHA = 0.0997
_CENTER_COLOR = 195.8

# Minimum correlation between the mask and the image's edges before the inversion is
# applied; the stamp measures 0.07-0.8 on every sample and a misplaced mask ~0.005.
_CENTER_MIN_EVIDENCE = 0.03

_INPAINT_RADIUS = 3


def watermark_mask(size: tuple[int, int]) -> Image.Image:
    """A single-channel mask (255 = inpaint) covering the bottom-right watermark box."""
    width, height = size
    mask = Image.new("L", (width, height), 0)
    x0 = max(0, width - WATERMARK_BOX_WIDTH)
    y0 = max(0, height - WATERMARK_BOX_HEIGHT)
    mask.paste(255, (x0, y0, width, height))
    return mask


def _center_mask(canvas: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Full-frame float mask (0-1) of the center overlay, or ``None`` if it is off-image."""
    x0 = width - _CENTER_MASK_ANCHOR_X
    y0 = height - _CENTER_MASK_ANCHOR_Y
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(width, x0 + canvas.shape[1]), min(height, y0 + canvas.shape[0])
    if xe <= xs or ye <= ys:
        return None
    mask = np.zeros((height, width), np.float32)
    mask[ys:ye, xs:xe] = canvas[ys - y0 : ye - y0, xs - x0 : xe - x0]
    return mask


def _center_evidence(pixels: np.ndarray, mask: np.ndarray) -> float | None:
    """Correlation between the mask's edges and the image's edges, signed so that the
    overlay's lightening of dark areas and darkening of bright areas both count as
    agreement. ``None`` when too little of the overlay is inside the image."""
    height, width = mask.shape
    gray = pixels.mean(axis=2)
    sign = np.sign(_CENTER_COLOR - cv2.GaussianBlur(gray, (0, 0), 4))
    image_edges = np.clip(gray - cv2.GaussianBlur(gray, (0, 0), 3), -10, 10) * sign
    mask_edges = mask - cv2.GaussianBlur(mask, (0, 0), 3)

    region = cv2.dilate((mask > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    region[max(0, height - WATERMARK_BOX_HEIGHT - 15) :, max(0, width - WATERMARK_BOX_WIDTH - 10) :] = False
    if region.sum() < 400:
        return None

    a = mask_edges[region] - mask_edges[region].mean()
    b = image_edges[region] - image_edges[region].mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denominator < 1e-6:
        return None
    return float((a * b).sum() / denominator)


def _remove_center_watermark(image: Image.Image, canvas: np.ndarray) -> Image.Image:
    pixels = np.asarray(image, dtype=np.float32)
    mask = _center_mask(canvas, image.width, image.height)
    if mask is None:
        return image

    evidence = _center_evidence(pixels, mask)
    if evidence is None or evidence < _CENTER_MIN_EVIDENCE:
        logger.info("center watermark not detected (evidence=%s)", evidence)
        return image

    alpha = (mask * _CENTER_ALPHA)[..., None]
    restored = (pixels - alpha * _CENTER_COLOR) / (1 - alpha)
    logger.info("center watermark removed (evidence=%.3f)", evidence)
    return Image.fromarray(np.clip(np.rint(restored), 0, 255).astype(np.uint8))


class WatermarkRemover:
    """Loads the LaMa model and the center-watermark mask once per process and reuses
    them across images."""

    def __init__(self) -> None:
        self._lama: Any | None = None
        try:
            self._lama = load_lama_cpu()
        except Exception:
            logger.warning(
                "Failed to load LaMa inpainting model; falling back to cv2 Telea for all images",
                exc_info=True,
            )

        self._center_canvas: np.ndarray | None = None
        canvas = cv2.imread(str(CENTER_MASK_PATH), cv2.IMREAD_GRAYSCALE)
        if canvas is None:
            logger.warning(
                "Failed to load center watermark mask from %s; center watermark will not be removed",
                CENTER_MASK_PATH,
            )
        else:
            self._center_canvas = canvas.astype(np.float32) / 255.0

    def remove(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        if self._center_canvas is not None:
            rgb = _remove_center_watermark(rgb, self._center_canvas)

        mask = watermark_mask(rgb.size)
        if self._lama is not None:
            try:
                return _run_lama(self._lama, rgb, mask)
            except Exception:
                logger.warning(
                    "LaMa inpainting failed on this image; falling back to cv2 Telea", exc_info=True
                )
        return _cv2_inpaint(rgb, mask)


def load_lama_cpu() -> Any:
    """Load the big-lama torchscript checkpoint on CPU.

    ``simple_lama_inpainting.SimpleLama`` calls ``torch.jit.load(model_path)`` without
    ``map_location``, and the published checkpoint embeds CUDA-tagged constants, so it
    raises even when ``torch.cuda.is_available()`` is False. We reuse the library's
    download/prep helpers but load the checkpoint ourselves with ``map_location="cpu"``.

    Also used by ``pipeline.warmup`` to bake the checkpoint into the Docker image at
    build time, so it's exposed rather than kept module-private.
    """
    from simple_lama_inpainting.models.model import LAMA_MODEL_URL
    from simple_lama_inpainting.utils import download_model

    model_path = download_model(LAMA_MODEL_URL)
    model = torch.jit.load(model_path, map_location="cpu")  # type: ignore[no-untyped-call]
    model.eval()
    return model


def _run_lama(model: Any, image: Image.Image, mask: Image.Image) -> Image.Image:
    from simple_lama_inpainting.utils import prepare_img_and_mask

    image_t, mask_t = prepare_img_and_mask(image, mask, torch.device("cpu"))
    with torch.inference_mode():
        inpainted = model(image_t, mask_t)
    result = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def _cv2_inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    arr = np.array(image)
    mask_arr = np.array(mask)
    out = cv2.inpaint(arr, mask_arr, _INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return Image.fromarray(out)
