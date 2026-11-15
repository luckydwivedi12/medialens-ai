"""Offline dataset audit CLI; run with the backend virtual environment."""

import argparse
import sys
from pathlib import Path

# Permit the documented invocation from the repository root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data.report import audit_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Read-only local media audit. Never downloads, repairs, extracts or trains.')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('D:/MediaLensData/reports'))
    parser.add_argument('--manifest', type=Path, help='Phase 2 CSV manifest')
    parser.add_argument('--extensions', nargs='+', help='Subset of supported extensions, space or comma separated')
    parser.add_argument('--sample-frames', type=int, default=3, help='Frames per video to sample (1-32; default 3)')
    args = parser.parse_args(argv)
    extensions = [part for value in args.extensions for part in value.split(',')] if args.extensions else None
    try:
        json_path, csv_path, report = audit_dataset(
            args.dataset_root, args.dataset_name, args.output_dir,
            args.manifest, extensions, args.sample_frames,
            progress=lambda message: print(message, file=sys.stderr),
        )
    except (OSError, ValueError) as exc:
        print(f'Audit could not complete: {exc}', file=sys.stderr)
        return 2
    print(f'JSON report: {json_path}\nCSV inventory: {csv_path}')
    return 1 if report['overall_status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
