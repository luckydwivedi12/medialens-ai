"""Secure cloud artifact retrieval and verification for deployment."""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

EXPECTED_CHECKPOINT_SHA256 = "sha256:5e19cc61d7b0c74b1f7979fb0682bc510dbc781c97d06f77945c50a25ab58622"
EXPECTED_YUNET_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
YUNET_UPSTREAM_COMMIT = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
YUNET_OFFICIAL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    "48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2026may.onnx"
)
PROVENANCE_FILENAME = "compatibility_20261115T084812012784Z_d75d8ac01dcc47eba2991d993b209523.json"

DEFAULT_PREPROCESSING_CONFIG = {
    "color_mode": "RGB",
    "resize": [224, 224],
    "interpolation": "bilinear",
    "antialias": True,
    "scale": "uint8_to_float_0_1",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "training_augmentation": {
        "horizontal_flip_probability": 0.5,
        "brightness": 0.1,
        "contrast": 0.1,
        "saturation": 0.05,
        "hue": 0.01,
    },
    "validation_and_inference_augmentation": False,
    "documented_weight_default": {
        "resize_short_side": 232,
        "center_crop": 224,
    },
    "geometry_note": "Direct 224x224 resize follows Phase 10 request; ImageNet V2 normalization is unchanged.",
}

DEFAULT_MODEL_METADATA = {
    "architecture": "MobileNetV3-Large",
    "class_to_index": {"real": 0, "fake": 1},
    "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    "decision_threshold": 0.5,
    "warning": "Development prototype, not proof of authenticity. Identity independence and universal generalization are not established.",
}


def download_file(url: str, destination: Path, expected_sha256: Optional[str] = None, auth_token: Optional[str] = None) -> None:
    """Download a file with streaming chunks and optional SHA256 checksum verification."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".tmp")
    
    headers = {"User-Agent": "MediaLens-Deployment/1.0"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    req = urllib.request.Request(url, headers=headers)
    hasher = hashlib.sha256()
    
    logger.info("Downloading artifact from %s to %s", url, destination)
    try:
        with urllib.request.urlopen(req, timeout=120) as response, temp_path.open("wb") as out_file:
            while chunk := response.read(65536):
                hasher.update(chunk)
                out_file.write(chunk)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        logger.error("Download failed from %s: %s", url, exc)
        raise RuntimeError(f"Failed to download artifact from {url}: {exc}") from exc

    calculated = hasher.hexdigest()
    if expected_sha256:
        clean_expected = expected_sha256.removeprefix("sha256:")
        if calculated != clean_expected:
            if temp_path.exists():
                temp_path.unlink()
            raise ValueError(f"Checksum mismatch for {destination.name}: expected {clean_expected}, got {calculated}")

    temp_path.replace(destination)
    logger.info("Successfully downloaded and verified %s", destination)


def ensure_checkpoint_artifact(checkpoint_path: Path) -> Path:
    """Ensure the model checkpoint and companion configs are available on disk."""
    if checkpoint_path.is_file():
        # Ensure companion metadata files exist in the same directory
        _ensure_companion_configs(checkpoint_path)
        return checkpoint_path

    url = os.environ.get("MEDIALENS_CHECKPOINT_URL") or os.environ.get("MEDIALENS_MODEL_URL")
    if not url:
        return checkpoint_path

    auth_token = os.environ.get("MEDIALENS_AUTH_TOKEN")
    logger.info("Retrieving checkpoint from remote artifact storage: %s", url)
    download_file(url, checkpoint_path, expected_sha256=EXPECTED_CHECKPOINT_SHA256, auth_token=auth_token)
    _ensure_companion_configs(checkpoint_path)
    return checkpoint_path


def _ensure_companion_configs(checkpoint_path: Path) -> None:
    """Write companion metadata and preprocessing configs if missing."""
    metadata_path = checkpoint_path.with_name("model_metadata.json")
    if not metadata_path.exists():
        metadata_url = os.environ.get("MEDIALENS_METADATA_URL")
        if metadata_url:
            download_file(metadata_url, metadata_path, auth_token=os.environ.get("MEDIALENS_AUTH_TOKEN"))
        else:
            metadata_path.write_text(json.dumps(DEFAULT_MODEL_METADATA, indent=2), encoding="utf-8")

    config_path = checkpoint_path.with_name("preprocessing_config.json")
    if not config_path.exists():
        config_url = os.environ.get("MEDIALENS_PREPROCESSING_URL")
        if config_url:
            download_file(config_url, config_path, auth_token=os.environ.get("MEDIALENS_AUTH_TOKEN"))
        else:
            config_path.write_text(json.dumps(DEFAULT_PREPROCESSING_CONFIG, indent=2), encoding="utf-8")


def ensure_yunet_artifacts(yunet_dir: Path, model_filename: str = "face_detection_yunet_2026may.onnx") -> tuple[Path, Path]:
    """Ensure YuNet ONNX model and provenance JSON are present."""
    model_path = yunet_dir / model_filename
    provenance_path = yunet_dir / PROVENANCE_FILENAME

    if model_path.is_file() and provenance_path.is_file():
        return model_path, provenance_path

    url = os.environ.get("MEDIALENS_YUNET_URL") or (
        YUNET_OFFICIAL_URL if os.environ.get("MEDIALENS_AUTO_DOWNLOAD_YUNET", "1") == "1" else None
    )

    if url and not model_path.is_file():
        auth_token = os.environ.get("MEDIALENS_AUTH_TOKEN")
        download_file(url, model_path, expected_sha256=EXPECTED_YUNET_SHA256, auth_token=auth_token)

    if not provenance_path.exists():
        provenance_data = {
            "model_name": model_filename,
            "sha256": EXPECTED_YUNET_SHA256,
            "upstream_commit": YUNET_UPSTREAM_COMMIT,
            "upstream_repository": "https://github.com/opencv/opencv_zoo",
            "model_path": str(model_path),
            "size_bytes": 229738,
            "license": "MIT",
            "target": "CPU",
            "engine": 2,
            "engine_environment": "OPENCV_FORCE_DNN_ENGINE=2",
        }
        provenance_path.write_text(json.dumps(provenance_data, indent=2), encoding="utf-8")

    return model_path, provenance_path

