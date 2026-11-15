"""Official ImageNet initialization and a frozen-feature MobileNetV3 head."""

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from app.data.inventory import sha256_file

WEIGHTS_URL = 'https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth'
PRETRAINED_SHA256 = '5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997'


def acquire_pretrained(directory: Path) -> tuple[Path, dict[str, Any]]:
    if MobileNet_V3_Large_Weights.IMAGENET1K_V2.url != WEIGHTS_URL:
        raise ValueError('Installed TorchVision weight URL differs from the requested official source')
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'mobilenet_v3_large-5c1a4163.pth'
    reused = target.exists()
    if not reused:
        temporary = directory / f'download_{uuid4().hex}.tmp'
        torch.hub.download_url_to_file(WEIGHTS_URL, str(temporary), hash_prefix='5c1a4163', progress=True)
        with temporary.open('rb') as source, target.open('xb') as output:
            shutil.copyfileobj(source, output)
        temporary.unlink()
    checksum = sha256_file(target)
    if checksum != 'sha256:' + PRETRAINED_SHA256:
        raise ValueError('Official pretrained weight full SHA-256 failed')
    return target, {'architecture': 'torchvision.models.mobilenet_v3_large',
        'weights': 'MobileNet_V3_Large_Weights.IMAGENET1K_V2', 'url': WEIGHTS_URL,
        'path': str(target), 'sha256': checksum, 'reused_existing_file': reused,
        'verification': 'Official published filename hash prefix and pinned full SHA-256 checked.'}


def build_model(pretrained_path: Path | None = None) -> nn.Module:
    model = mobilenet_v3_large(weights=None)
    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)
    model.features.eval()
    return model.cpu()


def frozen_features_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.features.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    """Serialize to an exclusively created file; existing checkpoints are sacred."""
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        with temporary.open('xb') as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic, same-volume publication; refuses existing target.
    finally:
        if temporary.exists():
            temporary.unlink()
