"""Shared CLIP loading/embedding helper.

Used by the offline classifier training tooling in ``scripts/`` and
``bikez_crawler.training``. The image pipeline is a separate, parallel piece of work and
does not import this yet — if it wants CLIP embeddings for inference, it should reuse
``ClipEmbedder``/``get_shared_embedder`` from here rather than re-implementing CLIP
loading, to stay compatible with the classifiers trained by this tooling.

Fixed contract with the trained classifier artifacts in ``models/*.joblib``: open_clip,
model ``ViT-B-32``, pretrained ``openai``, L2-normalized 512-dim image embeddings.
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import cast

import numpy as np
import open_clip
import torch
from PIL import Image

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
EMBED_DIM = 512


class ClipEmbedder:
    """Loads CLIP once (CPU) and embeds images/text with it."""

    def __init__(self) -> None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        model.eval()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        self.logit_scale: float = float(model.logit_scale.exp().item())

    @torch.inference_mode()
    def embed_image_bytes(self, data: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return cast(np.ndarray, self.embed_images([image])[0])

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Batch-embed already-loaded RGB images. Returns L2-normalized (n, 512) float32."""
        tensors = torch.stack([self.preprocess(img) for img in images])
        features = self.model.encode_image(tensors)
        features = features / features.norm(dim=-1, keepdim=True)
        return cast(np.ndarray, features.numpy().astype(np.float32))

    @torch.inference_mode()
    def embed_text(self, prompts: list[str]) -> np.ndarray:
        """Batch-embed text prompts. Returns L2-normalized (n, 512) float32."""
        tokens = self.tokenizer(prompts)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return cast(np.ndarray, features.numpy().astype(np.float32))


@lru_cache(maxsize=1)
def get_shared_embedder() -> ClipEmbedder:
    """Process-wide singleton so repeated calls don't reload CLIP weights."""
    return ClipEmbedder()
