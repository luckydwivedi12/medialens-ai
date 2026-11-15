"""Benchmark a saved checkpoint on a validation image; never load test pixels."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.dataset import load_development_records
from app.ml.inference import load_checkpoint
from app.ml.training import latency_benchmark, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split-report', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--threads', type=int, choices=(4, 6), required=True)
    args = parser.parse_args()
    if args.output_directory.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error('Benchmark output must remain outside the repository')
    torch.set_num_threads(args.threads)
    records, _ = load_development_records(args.split_report)
    model, _ = load_checkpoint(args.checkpoint)
    result = latency_benchmark(model, records['validation'][0])
    args.output_directory.mkdir(parents=True, exist_ok=True)
    path = args.output_directory / f'cpu_latency_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex}.json'
    write_json(path, result)
    print(json.dumps({'report': str(path), **result}, indent=2))


if __name__ == '__main__':
    main()
