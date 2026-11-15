"""Independently verify a frozen Phase 9 split and its prospective test lock."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data.dataset_quality import reconcile, recording_metadata, write_json
from app.data.inventory import sha256_file
from app.data.mask_curation import read_csv
from app.data.split_design import design_splits, validate_assignments
from app.data.visual_review import review_selections, sample_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    package = Path(report['outputs']['package'])
    checked: dict[str, str] = {}

    def check(path: Path, expected: str) -> None:
        key = str(path)
        if key not in checked:
            checked[key] = sha256_file(path)
        if checked[key] != expected:
            raise ValueError(f'File changed: {path}')

    for record in report['source_inputs'].values():
        check(Path(record['path']), record['checksum'])
    check(Path(report['phase8_report']), report['phase8_report_checksum'])
    for index in ('preservation_baseline.json', 'output_checksums.json'):
        for filename, expected in json.loads((package / index).read_text()).items():
            check(Path(filename), expected)
    original_report = json.loads(Path(report['phase8_report']).read_text())
    videos = read_csv(Path(report['source_inputs']['source_manifest']['path']))
    frames = read_csv(Path(report['source_inputs']['frame_manifest']['path']))
    source_detections = read_csv(Path(original_report['outputs']['mask_assisted_face_manifest']))
    pairs = read_csv(Path(original_report['outputs']['face_pair_manifest']))
    official = json.loads(Path(report['official_bundle']['path']).read_text())
    recording_evidence = report['official_recording_metadata']
    check(Path(recording_evidence['path']), recording_evidence['checksum'])
    recordings = recording_metadata(Path(recording_evidence['path']))
    plan = design_splits(videos, official['splits'], recordings=recordings['recordings'])
    if plan != report['split_plan']:
        # JSON converts edge tuples to lists; compare canonical serialized values.
        if json.dumps(plan, sort_keys=True) != json.dumps(report['split_plan'], sort_keys=True):
            raise ValueError('Recomputed split plan differs')
    counts, unmatched = reconcile(source_detections, pairs, frames)
    if counts != report['reconciliation']:
        raise ValueError('Count reconciliation differs')
    split_rows = read_csv(Path(report['outputs']['split_manifest']))
    assets = read_csv(Path(report['outputs']['asset_inventory']))
    def key(row: dict[str, str]) -> tuple[str, ...]:
        return tuple(row[k] for k in ('video_id', 'frame_index', 'detection_index', 'crop_path', 'checksum', 'status'))
    if Counter(key(r) for r in split_rows) != Counter(key(r) for r in source_detections):
        raise ValueError('Split manifest discarded or invented detection/failure records')
    for row in split_rows:
        if row['split'] != row['proposed_split'] or row['prior_split'] != 'unassigned':
            raise ValueError('Actual assignments differ from validated proposals')
        if row['training_approved'] != 'False' or row['manual_review_required'] != 'True':
            raise ValueError('Unreviewed candidates became approved training labels')
    for asset in assets:
        check(Path(asset['path']), asset['checksum'])
    validation = validate_assignments(videos, split_rows, assets, pairs, plan)
    if not validation['passed']:
        raise ValueError(f'Leakage validation failed: {validation["errors"]}')
    lock = json.loads(Path(report['test_lock']).read_text())
    if lock['status'] != 'locked' or sha256_file(Path(lock['split_manifest'])) != lock['split_manifest_checksum']:
        raise ValueError('Test lock is absent or split manifest changed')
    digest = hashlib.sha256(json.dumps(plan['component_splits'], sort_keys=True).encode()).hexdigest()
    if digest != lock['assignment_sha256'] or digest != report['assignment_sha256']:
        raise ValueError('Locked component membership changed')
    if Counter((a['path'], a['checksum'], a['component_id']) for a in lock['test_assets']) != Counter(
            (a['path'], a['checksum'], a['component_id']) for a in assets if a['split'] == 'test'):
        raise ValueError('Locked test assets differ')
    review = read_csv(Path(report['visual_review']['index_path']))
    asset_by_path = {a['path']: a for a in assets}
    for row in review:
        if row['split'] == 'test':
            raise ValueError('Test entered visual-review index')
        for name in ('primary_image', 'secondary_image'):
            if row[name] and asset_by_path[str(Path(row[name]).resolve())]['split'] == 'test':
                raise ValueError('Test pixels entered a development review sheet')
    selections = review_selections(split_rows, pairs)
    if Counter((category, sample_id(row)) for category, rows in selections.items() for row in rows) != Counter(
            (r['category'], r['sample_id']) for r in review):
        raise ValueError('Deterministic review selection differs')
    saved_unmatched = read_csv(package / 'unmatched_candidates.csv')
    if Counter(r['sample_id'] for r in saved_unmatched) != Counter(r['sample_id'] for r in unmatched):
        raise ValueError('Unmatched records were discarded')
    result = {'status': 'passed', 'report': str(args.report), 'files_checksum_verified': len(checked),
              'detection_failure_records': len(split_rows), 'asset_lineage_records': len(assets),
              'unmatched_candidates_preserved': len(saved_unmatched), 'visual_review_index_rows': len(review),
              'test_review_rows': 0, 'locked_test_asset_count': len(lock['test_assets']),
              'lineage_and_duplicate_validation': validation, 'assignment_sha256': digest}
    path = package / f'independent_verification_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex}.json'
    write_json(path, result)
    print(json.dumps({**result, 'verification_path': str(path)}, indent=2))


if __name__ == '__main__':
    main()
