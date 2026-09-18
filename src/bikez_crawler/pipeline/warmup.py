"""Pre-downloads model weights so they're baked into the Docker image at build time
instead of being fetched by the first container that starts in production.

Run standalone: ``python -m bikez_crawler.pipeline.warmup``. Unlike ``WatermarkRemover``,
this does not swallow a failed LaMa download -- a warmup that silently fails would bake an
empty cache into the image and just move the download (and its failure mode) to the first
real request instead.
"""

from __future__ import annotations

import open_clip
from rembg import new_session

from .background import REMBG_MODEL
from .classify import CLIP_MODEL_NAME, CLIP_PRETRAINED
from .watermark import load_lama_cpu


def main() -> None:
    load_lama_cpu()
    new_session(REMBG_MODEL)
    open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)


if __name__ == "__main__":
    main()
