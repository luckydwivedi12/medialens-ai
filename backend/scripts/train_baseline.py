"""Run at most five CPU-only frozen-feature classifier epochs on development data."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.training import TrainingConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-report', type=Path, required=True)
    parser.add_argument('--storage-root', type=Path, default=Path('D:/MediaLensData'))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, choices=(4, 8), default=8)
    parser.add_argument('--max-training-seconds', type=int, default=1800)
    parser.add_argument('--threads', type=int, choices=(4, 6), help='Reproduce a recorded thread setting; both settings are still benchmarked')
    args = parser.parse_args()
    run_training(args.split_report, args.storage_root, TrainingConfig(max_epochs=args.epochs,
        batch_size=args.batch_size, max_training_seconds=args.max_training_seconds, cpu_threads=args.threads))


if __name__ == '__main__':
    main()
