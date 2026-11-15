#!/usr/bin/env bash
# Build script for Render native Python environment
set -euo pipefail

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt -r requirements-ml.txt

echo "=== Ensuring model artifacts ==="
python scripts/download_models.py || true

echo "=== Build completed successfully ==="

