"""Read only frozen, matched-pair development crops; never open locked-test pixels."""

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.models import MobileNet_V3_Large_Weights

from app.data.inventory import sha256_file

CLASS_TO_INDEX = {'real': 0, 'fake': 1}
DEVELOPMENT_SPLITS = ('train', 'validation')


@dataclass(frozen=True)
class CropRecord:
    sample_id: str
    path: Path
    checksum: str
    label: int
    split: str
    component_id: str
    pair_id: str
    video_id: str
    frame_index: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def load_development_records(split_report: Path, verify_images: bool = True
                             ) -> tuple[dict[str, list[CropRecord]], dict[str, Any]]:
    """Test rows are inspected as metadata only; their paths are never statted/read."""
    report = json.loads(split_report.read_text(encoding='utf-8'))
    if report['assignments_status'] != 'assigned' or not report['split_validation']['passed']:
        raise ValueError('Phase 9 splits must be assigned and leakage validation must pass')
    manifest = Path(report['outputs']['split_manifest'])
    lock = json.loads(Path(report['test_lock']).read_text(encoding='utf-8'))
    if lock['status'] != 'locked' or sha256_file(manifest) != lock['split_manifest_checksum']:
        raise ValueError('Split manifest checksum or locked-test status differs')
    phase8_path = Path(report['phase8_report'])
    if sha256_file(phase8_path) != report['phase8_report_checksum']:
        raise ValueError('Phase 8 report checksum differs')
    phase8 = json.loads(phase8_path.read_text(encoding='utf-8'))
    package_checksums = json.loads((Path(report['outputs']['package']) / 'output_checksums.json').read_text())
    if package_checksums.get(str(split_report)) != sha256_file(split_report):
        raise ValueError('Frozen Phase 9 report checksum differs')
    pairs_path = Path(phase8['outputs']['face_pair_manifest'])
    # This CSV is part of the final Phase 9 preservation baseline.
    baseline = json.loads((Path(report['outputs']['package']) / 'preservation_baseline.json').read_text())
    if baseline.get(str(pairs_path)) != sha256_file(pairs_path):
        raise ValueError('Source crop-pair manifest changed')
    pairs = {f"{r['video_id']}:f{r['frame_index']}": r for r in read_csv(pairs_path)
             if r['status'] == 'matched_region_candidate'}
    rows = read_csv(manifest)
    component_splits: dict[str, set[str]] = defaultdict(set)
    selected: dict[str, list[dict[str, str]]] = defaultdict(list)
    hashes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row['split'] not in (*DEVELOPMENT_SPLITS, 'test'):
            raise ValueError('Unexpected split assignment')
        component_splits[row['component_id']].add(row['split'])
        if row['split'] != report['split_plan']['video_splits'].get(row['video_id']):
            raise ValueError('Video/component assignment changed')
        if row['component_id'] != report['split_plan']['component_for_video'].get(row['video_id']):
            raise ValueError('Video component changed')
        if row['crop_path']:
            hashes[row['checksum']].add(row['split'])
        if row['paired_subset_candidate'] == 'True':
            if row['selection_status'] != 'matched_region_pair_candidate' or row['status'] != 'crop_saved':
                raise ValueError('Unresolved/bystander/failure row entered paired selection')
            if row['label'] not in CLASS_TO_INDEX or not row['pair_id'] or not row['crop_path']:
                raise ValueError('Invalid paired label or crop path')
            selected[row['pair_id']].append(row)
    if any(len(s) > 1 for s in component_splits.values()) or any(len(s) > 1 for s in hashes.values()):
        raise ValueError('Cross-split group or duplicate-checksum leakage')
    result: dict[str, list[CropRecord]] = {s: [] for s in DEVELOPMENT_SPLITS}
    metadata_counts: Counter[str] = Counter()
    paired_components: dict[str, set[str]] = defaultdict(set)
    seen_paths: set[str] = set()
    root = split_report.parent.parent
    for pair_id, records in sorted(selected.items()):
        if len(records) != 2 or {r['label'] for r in records} != {'real', 'fake'}:
            raise ValueError('Expected exactly one original and one fake per matched pair')
        if len({(r['split'], r['component_id'], r['frame_index']) for r in records}) != 1:
            raise ValueError('Matched pair crosses partition, component or frame index')
        pair = pairs.get(pair_id)
        if not pair or not json.loads(pair['alignment_evidence'])['verified']:
            raise ValueError('Matched pair lacks verified alignment evidence')
        for row in records:
            prefix = 'fake' if row['label'] == 'fake' else 'original'
            expected_video = pair['video_id'] if prefix == 'fake' else pair['target_video_id']
            if (row['video_id'] != expected_video or row['crop_path'] != pair[f'{prefix}_crop_path'] or
                    row['checksum'] != pair[f'{prefix}_crop_checksum'] or row['frame_index'] != pair['frame_index']):
                raise ValueError('Matched crop path/checksum/video/frame differs')
            # Lexical traversal checks need no filesystem operation on a test path.
            lexical = PureWindowsPath(row['crop_path']) if '\\' in row['crop_path'] else Path(row['crop_path'])
            if '..' in lexical.parts or not lexical.is_absolute() or row['crop_path'] in seen_paths:
                raise ValueError('Invalid, traversing or duplicate selected crop path')
            seen_paths.add(row['crop_path'])
            metadata_counts[row['split']] += 1
            paired_components[row['split']].add(row['component_id'])
            if row['split'] == 'test':
                continue
            path = Path(row['crop_path']).resolve(strict=True)
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError('Development image escapes approved external data root')
            if verify_images:
                if sha256_file(path) != row['checksum']:
                    raise ValueError('Development crop checksum differs')
                with Image.open(path) as image:
                    image.verify()
            result[row['split']].append(CropRecord(
                sample_id=f"{row['video_id']}:f{row['frame_index']}:d{row['detection_index']}",
                path=path, checksum=row['checksum'], label=CLASS_TO_INDEX[row['label']], split=row['split'],
                component_id=row['component_id'], pair_id=pair_id, video_id=row['video_id'], frame_index=row['frame_index']))
    expected = {'train': 1172, 'validation': 300, 'test': 236}
    if dict(metadata_counts) != expected:
        raise ValueError(f'Paired counts differ; investigate instead of forcing counts: {dict(metadata_counts)}')
    return result, {'split_report': str(split_report), 'split_report_checksum': sha256_file(split_report),
        'split_manifest': str(manifest), 'split_manifest_checksum': sha256_file(manifest),
        'pair_manifest': str(pairs_path), 'pair_manifest_checksum': sha256_file(pairs_path),
        'class_to_index': CLASS_TO_INDEX, 'paired_image_counts': dict(metadata_counts),
        'paired_components': {s: len(g) for s, g in paired_components.items()},
        'all_component_counts': report['split_plan']['component_counts'],
        'development_images_checksum_verified': verify_images, 'test_images_opened': 0,
        'label_caveat': 'Matched region candidates remain provisional; Phase 10 explicitly authorizes their prototype use. Source review/approval fields are unchanged.'}


def preprocessing_config() -> dict[str, Any]:
    original = MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms()
    return {'color_mode': 'RGB', 'resize': [224, 224], 'interpolation': 'bilinear', 'antialias': True,
        'scale': 'uint8_to_float_0_1', 'mean': list(original.mean), 'std': list(original.std),
        'training_augmentation': {'horizontal_flip_probability': 0.5,
            'brightness': 0.1, 'contrast': 0.1, 'saturation': 0.05, 'hue': 0.01},
        'validation_and_inference_augmentation': False,
        'documented_weight_default': {'resize_short_side': 232, 'center_crop': 224},
        'geometry_note': 'Direct 224x224 resize follows Phase 10 request; ImageNet V2 normalization is unchanged.'}


def build_transform(training: bool, config: dict[str, Any] | None = None) -> T.Compose:
    config = config or preprocessing_config()
    operations: list[Any] = [T.Resize(tuple(config['resize']), interpolation=T.InterpolationMode.BILINEAR, antialias=config['antialias'])]
    if training:
        a = config['training_augmentation']
        operations.extend([T.RandomHorizontalFlip(a['horizontal_flip_probability']),
            T.ColorJitter(brightness=a['brightness'], contrast=a['contrast'], saturation=a['saturation'], hue=a['hue'])])
    operations.extend([T.ToTensor(), T.Normalize(config['mean'], config['std'])])
    return T.Compose(operations)


class MatchedCropDataset(Dataset[tuple[Tensor, int, int]]):
    def __init__(self, records: list[CropRecord], training: bool = False,
                 config: dict[str, Any] | None = None) -> None:
        if not records or any(r.split not in DEVELOPMENT_SPLITS or r.label not in (0, 1) for r in records):
            raise ValueError('Only nonempty, labeled development datasets are permitted in Phase 10')
        if training and any(r.split != 'train' for r in records):
            raise ValueError('Augmentation is restricted to training records')
        self.records, self.transform = records, build_transform(training, config)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        record = self.records[index]
        if record.split not in DEVELOPMENT_SPLITS:
            raise ValueError('Locked test image access prohibited')
        with Image.open(record.path) as source:
            tensor = self.transform(source.convert('RGB'))
        return tensor, record.label, index
