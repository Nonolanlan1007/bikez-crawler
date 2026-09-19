"""bikez.com watermark removal.

Both watermarks stamped on the poster images are alpha blends of a fixed graphic, so
each is undone exactly by inverting its blend: the pixels underneath are recovered
rather than repainted.

If the corner logo isn't found (e.g. a different logo version) that box is inpainted
instead. Inpainting uses LaMa (``simple_lama_inpainting``), which reconstructs textured
backgrounds far better than classical inpainting. ``cv2.inpaint`` (Telea) is the
fallback when LaMa can't be loaded or fails on a given image, since it's ~200x faster
but leaves a visible smear on non-flat backgrounds.
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

# sRGB blend: stamped = (1 - ALPHA * mask) * original + ALPHA * mask * COLOR. The mask
# PNG holds the "Bikez" letters and the plate behind ".com" in its red channel, and the
# near-white ".com" text on that plate, with its own opacity and colour, in its green.
_CENTER_ALPHA = 0.0997
_CENTER_COLOR = 195.8
_CENTER_COM_ALPHA = 0.073
_CENTER_COM_COLOR = 255.0

# Minimum correlation between the mask and the image's edges before the inversion is
# applied; the stamp measures 0.07-0.8 on every sample and a misplaced mask ~0.005.
_CENTER_MIN_EVIDENCE = 0.03

CORNER_LAYER_PATH = Path(__file__).resolve().parent / "assets" / "corner_watermark.npz"

# The corner logo is an exact 269x110 rectangle, 10px in from the bottom-right corner,
# that keeps half of the photo: stamped = _CORNER_KEEP * original + layer.
_CORNER_BOX_INSET = 10
_CORNER_KEEP = 127 / 255

# Minimum correlation between the logo layer's edges and the image's; the logo measures
# 0.80-0.93 on every sample and a shifted position <0.09.
_CORNER_MIN_EVIDENCE = 0.4

# Inverting the blend doubles the JPEG noise, so the box is lightly denoised.
_CORNER_DENOISE = (5, 30, 1.5)

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
    """Full-frame float masks (0-1, one channel per layer) of the center overlay, or
    ``None`` if it is off-image."""
    x0 = width - _CENTER_MASK_ANCHOR_X
    y0 = height - _CENTER_MASK_ANCHOR_Y
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(width, x0 + canvas.shape[1]), min(height, y0 + canvas.shape[0])
    if xe <= xs or ye <= ys:
        return None
    mask = np.zeros((height, width, canvas.shape[2]), np.float32)
    mask[ys:ye, xs:xe] = canvas[ys - y0 : ye - y0, xs - x0 : xe - x0]
    return mask


def _center_evidence(pixels: np.ndarray, mask: np.ndarray) -> float | None:
    """Correlation between the mask's edges and the image's edges, signed so that the
    overlay's lightening of dark areas and darkening of bright areas both count as
    agreement. ``None`` when too little of the overlay is inside the image."""
    height, width = mask.shape[:2]
    mask = mask.max(axis=2)
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

    letters = mask[..., 0:1] * _CENTER_ALPHA
    com = mask[..., 1:2] * _CENTER_COM_ALPHA
    restored = (pixels - letters * _CENTER_COLOR - com * _CENTER_COM_COLOR) / (1 - letters - com)
    logger.info("center watermark removed (evidence=%.3f)", evidence)
    return Image.fromarray(np.clip(np.rint(restored), 0, 255).astype(np.uint8))


def _remove_corner_logo(image: Image.Image, layer: np.ndarray) -> Image.Image | None:
    """The image with the corner logo blend inverted, or ``None`` if the logo isn't there."""
    box_height, box_width = layer.shape[:2]
    y1, x1 = image.height - _CORNER_BOX_INSET, image.width - _CORNER_BOX_INSET
    y0, x0 = y1 - box_height, x1 - box_width
    if y0 < 0 or x0 < 0:
        return None

    pixels = np.asarray(image, dtype=np.float32)
    box = pixels[y0:y1, x0:x1]

    layer_gray, box_gray = layer.mean(axis=2), box.mean(axis=2)
    a = layer_gray - cv2.GaussianBlur(layer_gray, (0, 0), 2)
    b = box_gray - cv2.GaussianBlur(box_gray, (0, 0), 2)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    evidence = float((a * b).sum() / denominator) if denominator > 1e-6 else 0.0
    if evidence < _CORNER_MIN_EVIDENCE:
        logger.info("corner logo not detected (evidence=%.3f)", evidence)
        return None

    restored = np.clip((box - layer) / _CORNER_KEEP, 0, 255)
    diameter, sigma_color, sigma_space = _CORNER_DENOISE
    restored = cv2.bilateralFilter(restored, diameter, sigma_color, sigma_space)
    out = pixels.copy()
    out[y0:y1, x0:x1] = restored
    logger.info("corner logo removed (evidence=%.3f)", evidence)
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


class WatermarkRemover:
    """Loads the LaMa model and the watermark assets once per process and reuses them
    across images."""

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
        try:
            with Image.open(CENTER_MASK_PATH) as mask_image:
                self._center_canvas = np.asarray(mask_image.convert("RGB"), dtype=np.float32)[..., :2] / 255.0
        except OSError:
            logger.warning(
                "Failed to load center watermark mask from %s; center watermark will not be removed",
                CENTER_MASK_PATH,
                exc_info=True,
            )

        self._corner_layer: np.ndarray | None = None
        try:
            with np.load(CORNER_LAYER_PATH) as assets:
                self._corner_layer = assets["layer"].astype(np.float32)
        except OSError:
            logger.warning(
                "Failed to load corner logo layer from %s; the corner logo will be inpainted",
                CORNER_LAYER_PATH,
                exc_info=True,
            )

    def remove(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        if self._center_canvas is not None:
            rgb = _remove_center_watermark(rgb, self._center_canvas)

        if self._corner_layer is not None:
            restored = _remove_corner_logo(rgb, self._corner_layer)
            if restored is not None:
                return restored

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
    return Image.fromarray(result[: image.height, : image.width])


def _cv2_inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    arr = np.array(image)
    mask_arr = np.array(mask)
    out = cv2.inpaint(arr, mask_arr, _INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return Image.fromarray(out)
