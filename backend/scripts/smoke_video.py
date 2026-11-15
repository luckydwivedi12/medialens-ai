"""Read-only validation-video smoke using the real checkpoint and live API/proxy."""
import argparse
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx

def sha(path: Path) -> str:
    return 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='http://127.0.0.1:8000')
    parser.add_argument('--frontend', default='http://localhost:5175')
    args = parser.parse_args()
    checkpoint = Path(r'D:\MediaLensData\models\medialens_mobilenetv3\best_model.pth')
    before = sha(checkpoint)
    metadata = json.loads(checkpoint.with_name('model_metadata.json').read_text())
    manifest = Path(metadata['dataset']['split_manifest'])
    assert sha(manifest) == metadata['dataset']['split_manifest_checksum']
    with manifest.open(newline='', encoding='utf-8-sig') as handle:
        row = next(r for r in csv.DictReader(handle) if r['split'] == 'validation'
                   and r['paired_subset_candidate'] == 'True' and r['status'] == 'crop_saved')
    # Only now resolve/open the selected development source; no test paths are accessed.
    root = Path(r'D:\MediaLensData\raw\faceforensics').resolve()
    video = (root / row['relative_path']).resolve()
    assert video.is_relative_to(root) and sha(video) == row['source_video_checksum']
    temp_root = Path(tempfile.gettempdir())
    before_temps = set(temp_root.glob('medialens-video-*'))
    outputs: dict[str, Any] = {}
    for label, base in [('backend', args.backend), ('frontend_proxy', args.frontend)]:
        with httpx.Client(base_url=base, timeout=240, trust_env=False) as client:
            assert client.get('/api/v1/model/status').json()['video_ready']
            with video.open('rb') as handle:
                response = client.post('/api/v1/detect/video', files={'file': ('validation.mp4', handle, 'video/mp4')})
            response.raise_for_status()
            result = response.json()
            frames = [r for r in result['frame_level_results'] if r['status'] == 'analyzed']
            assert len(frames) >= 3 and result['status'] == 'analyzed'
            for frame in frames:
                scores = [f['probabilities']['fake'] for f in frame['faces'] if f['status'] == 'analyzed']
                assert abs(frame['average_fake_score'] - sum(scores)/len(scores)) < 1e-7
            assert abs(result['average_fake_score'] - sum(f['average_fake_score'] for f in frames)/len(frames)) < 1e-7
            outputs[label] = result
    a, b = outputs.values()
    assert a['frame_level_results'] == b['frame_level_results'] and a['average_fake_score'] == b['average_fake_score']
    # Independent CPU forward over the exact decoded regions returned by the API.
    import cv2
    import torch
    from PIL import Image
    from app.ml.dataset import build_transform
    from app.ml.inference import load_checkpoint
    torch.set_num_threads(6)
    model, saved = load_checkpoint(checkpoint)
    transform = build_transform(False, saved['preprocessing'])
    capture = cv2.VideoCapture(str(video))
    independent = []
    try:
        with torch.inference_mode():
            for frame in a['frame_level_results']:
                if frame['status'] != 'analyzed':
                    continue
                assert capture.set(cv2.CAP_PROP_POS_FRAMES, frame['frame_index'])
                ok, pixels = capture.read()
                assert ok
                face_scores = []
                for face in frame['faces']:
                    if face['status'] != 'analyzed':
                        continue
                    left, top, right, bottom = face['crop_box']
                    rgb = cv2.cvtColor(pixels[top:bottom, left:right], cv2.COLOR_BGR2RGB)
                    logits = model(transform(Image.fromarray(rgb)).unsqueeze(0))
                    scores = logits.softmax(dim=1)[0]
                    for name, index in saved['class_to_index'].items():
                        assert abs(float(scores[index]) - face['probabilities'][name]) < 1e-6
                    face_scores.append(float(scores[saved['class_to_index']['fake']]))
                independent.append(sum(face_scores) / len(face_scores))
    finally:
        capture.release()
    aggregate = sum(independent) / len(independent)
    assert abs(aggregate - a['average_fake_score']) < 1e-6
    assert a['predicted_label'] == ('fake' if aggregate >= a['threshold'] else 'real')
    assert set(temp_root.glob('medialens-video-*')) == before_temps
    assert sha(video) == row['source_video_checksum']
    assert sha(checkpoint) == before
    print(json.dumps({'video_id': row['video_id'], 'split': 'validation', 'checkpoint_unchanged': True,
        'locked_test_media_opened': 0, 'temporary_video_directories_cleaned': True, 'source_unchanged': True, 'proxy_matches_backend': True, 'results': outputs}, indent=2))

if __name__ == '__main__':
    main()
