"""Build and verify complete reciprocal-pair manifests without changing media."""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data.inventory import sha256_file
from app.data.models import MANIFEST_COLUMNS

SOURCE_URL = 'https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md'
EXTRA_COLUMNS = ('video_id', 'relative_path', 'target_video_id', 'donor_video_id',
                 'manipulation_method', 'leakage_group_id', 'sample_purpose', 'access_evidence')
ORIGINAL_DIR = Path('original_sequences/youtube/c23/videos')
FAKE_DIR = Path('manipulated_sequences/Deepfakes/c23/videos')


def build_manifest(
    root: Path, inventory: Path, *, access_date: str = '',
    access_approved: str = 'unknown', terms_reference: str = '', access_evidence: str = '',
    expected_groups: int = 10,
) -> list[dict[str, str]]:
    """Fail on invalid counts, names, relationships or hashes; never repair them."""
    if expected_groups < 1:
        raise ValueError('expected_groups must be positive')
    expected_count = expected_groups * 4
    if access_approved not in {'true', 'false', 'unknown'}:
        raise ValueError('Invalid access_approved assertion')
    if access_date and date.fromisoformat(access_date).isoformat() != access_date:
        raise ValueError('access_date must be YYYY-MM-DD')
    root = root.resolve(strict=True)
    with inventory.open(newline='', encoding='utf-8-sig') as handle:
        entries = list(csv.DictReader(handle))
    if len(entries) != expected_count:
        raise ValueError(f'Expected {expected_count} inventory records, found {len(entries)}')
    rows: list[dict[str, str]] = []
    expected_paths: set[Path] = set()
    originals: set[str] = set()
    fakes: dict[str, tuple[str, str]] = {}
    for entry in entries:
        path = Path(entry['path'])
        if path.is_symlink() or path.is_junction():
            raise ValueError(f'Linked media is not allowed: {path}')
        path = path.resolve(strict=True)
        relative = path.relative_to(root)
        if path in expected_paths or not path.is_file():
            raise ValueError(f'Duplicate path or non-file: {path}')
        expected_paths.add(path)
        if entry['valid'] != 'True' or entry['kind'] != 'video' or entry['extension'] != '.mp4':
            raise ValueError(f'Inventory record is not a valid MP4 video: {path}')
        if path.suffix != '.mp4' or path.stat().st_size != int(entry['size_bytes']):
            raise ValueError(f'File type or size differs from inventory: {path}')
        checksum = sha256_file(path)
        if checksum != entry['checksum']:
            raise ValueError(f'Checksum differs from inventory: {path}')
        video_id = path.stem
        row = {**dict.fromkeys(MANIFEST_COLUMNS + EXTRA_COLUMNS, ''),
               'dataset_name': 'faceforensics-sample', 'dataset_version': 'FaceForensics++-Deepfakes-c23',
               'source_url': SOURCE_URL, 'access_date': access_date, 'access_approved': access_approved,
               'terms_reference': terms_reference, 'video_id': video_id,
               'relative_path': relative.as_posix(), 'source_video_id': video_id,
               'split': 'unassigned', 'checksum': checksum, 'sample_purpose': 'pipeline_smoke_test',
               'access_evidence': access_evidence}
        if relative.parent == ORIGINAL_DIR and re.fullmatch(r'[0-9]{3}', video_id):
            originals.add(video_id)
            row.update(original_video_id=video_id, target_video_id=video_id,
                       manipulation_method='none', label='real')
        elif relative.parent == FAKE_DIR and re.fullmatch(r'[0-9]{3}_[0-9]{3}', video_id):
            target, donor = video_id.split('_')
            if target == donor:
                raise ValueError(f'Self-pair is not a reciprocal sample pair: {video_id}')
            fakes[video_id] = target, donor
            row.update(original_video_id=target, target_video_id=target,
                       donor_video_id=donor, manipulation_method='Deepfakes', label='fake')
        else:
            raise ValueError(f'Unexpected sample path/name: {relative}')
        rows.append(row)
    actual_paths = {p.resolve() for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.mp4'}
    if actual_paths != expected_paths:
        raise ValueError(f'Disk MP4 inventory differs from the {expected_count} supplied records')
    if len(originals) != expected_groups * 2 or len(fakes) != expected_groups * 2 or len({r['video_id'] for r in rows}) != expected_count:
        raise ValueError(f'Expected {expected_groups * 2} unique originals and {expected_groups * 2} unique manipulated videos')
    group_for: dict[str, str] = {}
    for video_id, (target, donor) in fakes.items():
        if target not in originals or donor not in originals:
            raise ValueError(f'Missing referenced original for {video_id}')
        if f'{donor}_{target}' not in fakes:
            raise ValueError(f'Missing reciprocal manipulated video for {video_id}')
        group_id = 'ffpp-c23-pair-' + '-'.join(sorted((target, donor)))
        for member in (target, donor, video_id):
            if member in group_for and group_for[member] != group_id:
                raise ValueError(f'Overlapping reciprocal pairs: {member}')
            group_for[member] = group_id
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row['video_id'] not in group_for:
            raise ValueError(f'Original has no reciprocal pair: {row["video_id"]}')
        row['leakage_group_id'] = group_for[row['video_id']]
        groups[row['leakage_group_id']].append(row)
    if len(groups) != expected_groups or any(Counter(r['label'] for r in group) != {'real': 2, 'fake': 2} for group in groups.values()):
        raise ValueError(f'Expected {expected_groups} groups with two real and two fake videos each')
    return sorted(rows, key=lambda row: (row['leakage_group_id'], row['label'], row['video_id']))


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS + EXTRA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--access-date', default='')
    parser.add_argument('--expected-groups', type=int, default=10)
    parser.add_argument('--access-approved', choices=['true', 'false', 'unknown'], default='unknown')
    parser.add_argument('--terms-reference', default='')
    parser.add_argument('--access-evidence', default='', help='User-supplied authorization provenance; no fabricated certificates')
    args = parser.parse_args()
    try:
        if args.output.resolve().is_relative_to(args.dataset_root.resolve()):
            raise ValueError('Manifest output must be outside the raw dataset root')
        rows = build_manifest(args.dataset_root, args.inventory, access_date=args.access_date,
                              access_approved=args.access_approved, terms_reference=args.terms_reference,
                              access_evidence=args.access_evidence, expected_groups=args.expected_groups)
        write_manifest(rows, args.output)
    except (OSError, ValueError, KeyError) as exc:
        print(f'Manifest not created: {exc}', file=sys.stderr)
        return 1
    print(f'Created {args.output}: {len(rows)} records, {args.expected_groups} verified reciprocal groups; split=unassigned')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
