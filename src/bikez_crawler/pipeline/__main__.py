"""Entrypoint: ``python -m bikez_crawler.pipeline --stage STAGE [--bike TAG]``."""

from __future__ import annotations

import argparse
import logging

from bikez_crawler.config import load_settings

from .worker import run


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU image-processing pipeline for bikez-crawler")
    parser.add_argument(
        "--stage",
        choices=("preprocess", "background"),
        default="preprocess",
        help="Run the model-isolated preprocessing or background-removal stage",
    )
    parser.add_argument("--bike", default=None, help="Restrict to a single bike tag (test mode)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = load_settings()
    if args.bike:
        settings.pipeline_bike_tag = args.bike

    run(settings, stage=args.stage)


if __name__ == "__main__":
    main()
