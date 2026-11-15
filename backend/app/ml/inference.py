"""Checkpoint-only CPU inference on development records with shared preprocessing."""

from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import nn
from PIL import Image

from .dataset import CLASS_TO_INDEX, DEVELOPMENT_SPLITS, CropRecord, MatchedCropDataset, preprocessing_config
from .model import build_model


class ImageInferenceService:
    """One CPU model per application lifetime; serialize CPU forwards."""

    def __init__(self, checkpoint_path: Path) -> None:
        import hashlib
        import json
        import os
        from threading import Lock

        from .dataset import build_transform

        metadata_path = checkpoint_path.with_name('model_metadata.json')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        checksum = 'sha256:' + hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if checksum != metadata['checkpoint_sha256']:
            raise ValueError('Checkpoint checksum differs from saved metadata')
        self.model, checkpoint = load_checkpoint(checkpoint_path)
        if metadata['class_to_index'] != checkpoint['class_to_index']:
            raise ValueError('Saved metadata class mapping differs')
        saved_config = json.loads(checkpoint_path.with_name('preprocessing_config.json').read_text())
        if saved_config != checkpoint['preprocessing']:
            raise ValueError('Saved preprocessing configuration differs')
        self.threshold = float(metadata.get('decision_threshold', .5))
        if not 0 < self.threshold < 1:
            raise ValueError('Invalid decision threshold')
        self.class_to_index = checkpoint['class_to_index']
        self.transform = build_transform(False, checkpoint['preprocessing'])
        self.version = f"phase10-epoch{checkpoint['epoch']}-{checksum[7:19]}"
        self.lock = Lock()
        num_threads = int(os.environ.get("TORCH_NUM_THREADS", min(6, os.cpu_count() or 1)))
        torch.set_num_threads(max(1, num_threads))

    def predict(self, image: Image.Image) -> dict[str, Any]:
        from app.schemas.detection import WARNING

        with self.lock, torch.inference_mode():
            tensor = self.transform(image).unsqueeze(0)
            logits = self.model(tensor)
            if logits.shape != (1, 2) or not torch.isfinite(logits).all():
                raise ValueError('Invalid classifier output')
            probabilities = torch.softmax(logits, dim=1)[0]
            scores = {label: float(probabilities[index]) for label, index in self.class_to_index.items()}
        return {'success': True, 'predicted_label': 'fake' if scores['fake'] >= self.threshold else 'real',
                'probabilities': scores, 'threshold': self.threshold, 'model_name': 'MobileNetV3-Large',
                'model_version': self.version, 'device': 'cpu', 'warning': WARNING}


def load_checkpoint(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1 or checkpoint.get('architecture') != 'mobilenet_v3_large':
        raise ValueError('Unsupported checkpoint format or architecture')
    if checkpoint.get('class_to_index') != CLASS_TO_INDEX:
        raise ValueError('Checkpoint class mapping differs')
    if checkpoint.get('preprocessing') != preprocessing_config():
        raise ValueError('Checkpoint preprocessing differs from the frozen prototype contract')
    model = build_model()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.cpu().eval(), checkpoint


def predict_records(model: nn.Module, records: list[CropRecord], config: dict[str, Any]
                     ) -> list[dict[str, Any]]:
    if any(r.split not in DEVELOPMENT_SPLITS for r in records):
        raise ValueError('Locked test inference is prohibited during Phase 10')
    dataset = MatchedCropDataset(records, training=False, config=config)
    model.eval()
    result: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, record in enumerate(records):
            started = perf_counter()
            tensor, _, _ = dataset[index]
            logits = model(tensor.unsqueeze(0))
            if logits.shape != (1, 2) or not torch.isfinite(logits).all():
                raise ValueError('Invalid binary classifier output')
            probabilities = torch.softmax(logits, dim=1)[0]
            if not torch.allclose(probabilities.sum(), torch.tensor(1.), atol=1e-6):
                raise ValueError('Softmax probabilities do not sum to one')
            result.append({'sample_id': record.sample_id, 'split': record.split, 'label_index': record.label,
                'predicted_index': int(probabilities[1] >= .5), 'p_real': float(probabilities[0]),
                'p_fake': float(probabilities[1]), 'logits': logits[0].tolist(),
                'end_to_end_seconds': perf_counter() - started,
                'interpretation': 'Prototype score, not proof of authenticity or calibrated probability.'})
    return result
