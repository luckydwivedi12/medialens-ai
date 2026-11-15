"""Audit official masks and existing face detections without assigning splits."""

import argparse
import base64
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data.inventory import sha256_file
from app.data.mask_association import AssociationThresholds
from app.data.mask_curation import curate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--faces', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--storage-root', type=Path, required=True)
    parser.add_argument('--official-splits', type=Path, help='Previously frozen official split bundle; otherwise fetch official GitHub')
    parser.add_argument('--foreground', type=int, default=127)
    parser.add_argument('--minimum-face-overlap', type=float, default=0.5)
    parser.add_argument('--minimum-mask-coverage', type=float, default=0.05)
    parser.add_argument('--original-iou', type=float, default=0.5)
    parser.add_argument('--thresholds-reviewed', action='store_true', help='Only after recording visual and sensitivity inspection in project documentation')
    args = parser.parse_args()
    storage = args.storage_root.resolve()
    if storage.is_relative_to(Path(__file__).resolve().parents[2]) or storage.is_relative_to(args.dataset_root.resolve()):
        parser.error('Storage must be outside repository and raw dataset')
    provenance: dict[str, object] = {}
    if args.official_splits:
        bundle = json.loads(args.official_splits.read_text(encoding='utf-8'))
        official = bundle['splits']
        provenance = {'path': str(args.official_splits), 'checksum': sha256_file(args.official_splits)}
    else:
        official = {}
        urls = {}
        for name in ('train', 'val', 'test'):
            url = f'https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{name}.json'
            try:
                with urllib.request.urlopen(url, timeout=12) as response:
                    official[name] = json.load(response)
                urls[name] = {'url': url}
            except OSError:
                api_url = f'https://api.github.com/repos/ondyari/FaceForensics/contents/dataset/splits/{name}.json'
                with urllib.request.urlopen(api_url, timeout=30) as response:
                    payload = json.load(response)
                official[name] = json.loads(base64.b64decode(payload['content']))
                urls[name] = {'url': api_url, 'git_blob_sha': payload['sha']}
        path = storage / 'reports' / f'official_splits_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as handle:
            json.dump({'urls': urls, 'splits': official}, handle, indent=2)
        provenance = {'path': str(path), 'checksum': sha256_file(path)}
    print('Official split evidence:', json.dumps(provenance), flush=True)
    report = curate(args.manifest, args.frames, args.faces, args.dataset_root, storage, official,
        AssociationThresholds(args.foreground, args.minimum_face_overlap, args.minimum_mask_coverage, args.original_iou),
        args.thresholds_reviewed, provenance)
    print(json.dumps({k: report[k] for k in ('report_path', 'valid_masks', 'aligned_masks',
        'frame_association_counts', 'pair_status_counts', 'leakage_audit')}, indent=2))


if __name__ == '__main__':
    main()
