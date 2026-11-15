"""Reconcile Phase 8, render development review sheets and freeze lineage splits."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data.dataset_quality import run_quality_control


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase8-report', type=Path, required=True)
    parser.add_argument('--storage-root', type=Path, default=Path('D:/MediaLensData'))
    parser.add_argument('--verify-official-live', action='store_true', help='Verify frozen official content through immutable GitHub blobs')
    parser.add_argument('--recording-metadata', type=Path, help='Frozen official recording metadata bundle for offline reproduction')
    args = parser.parse_args()
    result = run_quality_control(args.phase8_report, args.storage_root, args.verify_official_live, args.recording_metadata)
    return 0 if result['split_validation']['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
