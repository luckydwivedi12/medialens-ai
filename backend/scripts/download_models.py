"""Standalone script to retrieve and verify model weights for deployment."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.model_fetcher import (
    ensure_checkpoint_artifact,
    ensure_yunet_artifacts,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_YUNET_SHA256,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_models")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify MediaLens AI model artifacts for deployment.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Root directory where models should be stored (default: ./models)",
    )
    parser.add_argument("--checkpoint-url", type=str, default=None, help="Remote URL for best_model.pth")
    parser.add_argument("--yunet-url", type=str, default=None, help="Remote URL for YuNet ONNX")
    args = parser.parse_args()

    models_dir = args.models_dir.resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Checkpoint setup
    checkpoint_dir = models_dir / "medialens_mobilenetv3"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pth"

    if args.checkpoint_url:
        os.environ["MEDIALENS_CHECKPOINT_URL"] = args.checkpoint_url

    logger.info("Resolving classifier checkpoint at: %s", checkpoint_path)
    result_path = ensure_checkpoint_artifact(checkpoint_path)
    if result_path.is_file():
        logger.info("Classifier checkpoint verified and ready at: %s", result_path)
    else:
        logger.warning(
            "Classifier checkpoint best_model.pth was not found or downloaded. "
            "Set MEDIALENS_CHECKPOINT_URL to retrieve it during startup or build."
        )

    # 2. YuNet setup
    yunet_dir = models_dir / "yunet"
    yunet_dir.mkdir(parents=True, exist_ok=True)
    if args.yunet_url:
        os.environ["MEDIALENS_YUNET_URL"] = args.yunet_url

    logger.info("Resolving YuNet detector model at: %s", yunet_dir)
    try:
        model_path, provenance_path = ensure_yunet_artifacts(yunet_dir)
        logger.info("YuNet detector verified at: %s", model_path)
    except Exception as exc:
        logger.warning("YuNet download failed: %s", exc)


if __name__ == "__main__":
    main()

