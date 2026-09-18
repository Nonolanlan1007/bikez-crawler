"""Bottom-right bikez.com watermark removal.

Primary method is LaMa inpainting (``simple_lama_inpainting``), which reconstructs
textured backgrounds (roads, foliage, studio gradients) far better than classical
inpainting. ``cv2.inpaint`` (Telea) is the fallback when LaMa can't be loaded or fails
on a given image, since it's ~200x faster but leaves a visible smear on non-flat
backgrounds.
"""

from __future__ import annotations

import logging
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

_INPAINT_RADIUS = 3


def watermark_mask(size: tuple[int, int]) -> Image.Image:
    """A single-channel mask (255 = inpaint) covering the bottom-right watermark box."""
    width, height = size
    mask = Image.new("L", (width, height), 0)
    x0 = max(0, width - WATERMARK_BOX_WIDTH)
    y0 = max(0, height - WATERMARK_BOX_HEIGHT)
    mask.paste(255, (x0, y0, width, height))
    return mask


class WatermarkRemover:
    """Loads the LaMa model once per process and reuses it across images."""

    def __init__(self) -> None:
        self._lama: Any | None = None
        try:
            self._lama = load_lama_cpu()
        except Exception:
            logger.warning(
                "Failed to load LaMa inpainting model; falling back to cv2 Telea for all images",
                exc_info=True,
            )

    def remove(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
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
