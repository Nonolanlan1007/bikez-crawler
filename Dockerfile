# opencv-python (pulled in transitively by simple-lama-inpainting) needs these X11/GL
# shared libraries at import time even though we only ever use it headlessly; the build
# stage needs them too since the warmup step below imports the watermark-removal module.
ARG OPENCV_RUNTIME_DEPS="libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1"

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
ARG OPENCV_RUNTIME_DEPS
RUN apt-get update && apt-get install -y --no-install-recommends $OPENCV_RUNTIME_DEPS \
    && rm -rf /var/lib/apt/lists/*
# Same cache paths in both stages: the build stage populates them by loading each model
# once (see pipeline/warmup.py), and the runtime stage inherits the populated cache
# instead of every container re-downloading LaMa/BiRefNet's weights on first use.
ENV TORCH_HOME=/app/.cache/torch
ENV REMBG_HOME=/app/.cache/rembg
ENV HF_HOME=/app/.cache/huggingface
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
RUN uv run --no-dev python -m bikez_crawler.pipeline.warmup

FROM python:3.12-slim AS runtime
WORKDIR /app
ARG OPENCV_RUNTIME_DEPS
RUN apt-get update && apt-get install -y --no-install-recommends $OPENCV_RUNTIME_DEPS \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/app/.venv/bin:$PATH"
ENV TORCH_HOME=/app/.cache/torch
ENV REMBG_HOME=/app/.cache/rembg
ENV HF_HOME=/app/.cache/huggingface
COPY --from=build /app/.venv ./.venv
COPY --from=build /app/src ./src
COPY --from=build /app/.cache ./.cache
COPY models ./models
ENTRYPOINT ["python", "-m"]
CMD ["bikez_crawler.crawl"]
