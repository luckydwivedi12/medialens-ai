"""Read-only Phase 11 smoke: one validation crop, real checkpoint, live APIs.

Never resolves, stats, hashes or opens locked-test image paths. No uploads saved.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import torch
from PIL import Image

from app.ml.dataset import build_transform
from app.ml.inference import ImageInferenceService, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(r'D:\MediaLensData\models\medialens_mobilenetv3\best_model.pth'))
    parser.add_argument('--backend', default='http://127.0.0.1:8000')
    parser.add_argument('--frontend', default='http://localhost:5175')
    args = parser.parse_args()
    before = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    service = ImageInferenceService(args.checkpoint)
    model, checkpoint = load_checkpoint(args.checkpoint)
    manifest = Path(checkpoint['dataset_provenance']['split_manifest'])
    expected_manifest = checkpoint['dataset_provenance']['split_manifest_checksum']
    assert 'sha256:' + hashlib.sha256(manifest.read_bytes()).hexdigest() == expected_manifest
    with manifest.open(newline='', encoding='utf-8-sig') as handle:
        # Filter metadata before performing ANY image filesystem access.
        row = next(row for row in csv.DictReader(handle) if row['split'] == 'validation'
                   and row['paired_subset_candidate'] == 'True' and row['status'] == 'crop_saved')
    path = Path(row['crop_path'])
    content = path.read_bytes()
    assert 'sha256:' + hashlib.sha256(content).hexdigest() == row['checksum']
    with Image.open(path) as source:
        # Independently reproduce Phase 10 dataset inference, without service preprocessing.
        tensor = build_transform(False, checkpoint['preprocessing'])(source.convert('RGB'))
    with torch.inference_mode():
        scores = torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0]
    sample_id = f"{row['video_id']}:f{row['frame_index']}:d{row['detection_index']}"
    outputs: dict[str, Any] = {}
    for label, base in [('backend', args.backend), ('frontend_proxy', args.frontend)]:
        with httpx.Client(base_url=base, timeout=60, trust_env=False) as client:
            assert client.get('/api/health').json()['status'] == 'ok'
            status = client.get('/api/v1/model/status').json()
            assert status['model_ready'] and status['model_version'] == service.version
            response = client.post('/api/v1/detect/image', files={'file': ('validation.png', content, 'image/png')})
            response.raise_for_status()
            result = response.json()
            for name, index in checkpoint['class_to_index'].items():
                assert abs(result['probabilities'][name] - float(scores[index])) < 1e-6
            assert abs(sum(result['probabilities'].values()) - 1) < 1e-6
            repeated = client.post('/api/v1/detect/image', files={'file': ('validation.png', content, 'image/png')})
            assert repeated.json() == result
            outputs[label] = result
    assert httpx.get(args.frontend, trust_env=False).status_code == 200
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == before
    print(json.dumps({'validation_sample_id': sample_id, 'checkpoint_sha256': before,
        'checkpoint_unchanged': True, 'independent_phase10_probabilities_match': True,
        'repeated_predictions_identical': True, 'test_images_opened': 0, 'results': outputs}, indent=2))


if __name__ == '__main__':
    main()
