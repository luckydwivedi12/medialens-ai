"""Independently verify Phase 8 output closure and unchanged source assets."""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data.inventory import sha256_file
from app.data.mask_curation import read_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--preservation-baseline', type=Path, help='Optional earlier path-to-SHA256 preservation record')
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    checked: dict[str, str] = {}
    baseline_count = 0

    def check(path: Path, checksum: str) -> None:
        if str(path) not in checked:
            checked[str(path)] = sha256_file(path)
        if checked[str(path)] != checksum:
            raise ValueError(f'Checksum mismatch: {path}')

    for name in ('source_manifest', 'frame_manifest', 'face_manifest'):
        check(Path(report[name]), report[name + '_checksum'])
    if args.preservation_baseline:
        baseline = json.loads(args.preservation_baseline.read_text(encoding='utf-8'))
        for filename, checksum in baseline.items():
            check(Path(filename), checksum if checksum.startswith('sha256:') else 'sha256:' + checksum)
        baseline_count = len(baseline)
    check(Path(report['official_split_evidence']['path']), report['official_split_evidence']['checksum'])
    videos = read_csv(Path(report['source_manifest']))
    frames = read_csv(Path(report['frame_manifest']))
    faces = read_csv(Path(report['face_manifest']))
    for row in videos:
        check(args.dataset_root / row['relative_path'], row['checksum'])
    for row in frames:
        if row['frame_path']:
            check(Path(row['frame_path']), row['frame_checksum'])
    for row in faces:
        if row['crop_path']:
            check(Path(row['crop_path']), row['checksum'])
    masks = read_csv(Path(report['outputs']['mask_validation_manifest']))
    associations = read_csv(Path(report['outputs']['mask_assisted_face_manifest']))
    pairs = read_csv(Path(report['outputs']['face_pair_manifest']))
    for row in masks:
        if row['checksum']:
            check(Path(row['mask_path']), row['checksum'])
    def key(row: dict[str, str]) -> tuple[str, ...]:
        return tuple(row[k] for k in ('video_id', 'frame_index', 'detection_index', 'crop_path', 'status'))
    if Counter(key(r) for r in associations) != Counter(key(r) for r in faces):
        raise ValueError('Detection/failure record closure differs')
    if len(pairs) != sum(f['label'] == 'fake' for f in frames):
        raise ValueError('Not every manipulated frame has a pair/failure record')
    for row in pairs:
        if row['status'] == 'matched_region_candidate':
            if not json.loads(row['alignment_evidence'])['verified']:
                raise ValueError('Matched pair lacks temporal/content evidence')
            check(Path(row['fake_crop_path']), row['fake_crop_checksum'])
            check(Path(row['original_crop_path']), row['original_crop_checksum'])
        elif row['original_crop_path']:
            raise ValueError('Unresolved pair has an invented original crop')
    if any(r['split'] != 'unassigned' for r in videos + frames + faces + associations + pairs):
        raise ValueError('Unexpected split allocation')
    result = {'report': str(args.report), 'status': 'passed',
              'source_assets_checksum_verified': len(checked),
              'preservation_baseline_files': baseline_count,
              'mask_count': len(masks), 'detection_records_preserved': len(associations),
              'pair_frame_records': len(pairs),
              'zero_face_failure_records': sum(r['status'] == 'no_face' for r in associations),
              'ambiguous_face_records': sum(r['association_status'] == 'ambiguous_multiple_overlap' and bool(r['crop_path']) for r in associations),
              'split_values': ['unassigned']}
    path = args.report.parent / f'mask_curation_verification_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex}.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({**result, 'verification_path': str(path)}, indent=2))


if __name__ == '__main__':
    main()
