"""Sample local videos and localize faces; never downloads or classifies deepfakes."""

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--storage-root', type=Path, default=Path('D:/MediaLensData'))
    parser.add_argument('--model', type=Path, default=Path('D:/MediaLensData/models/yunet/face_detection_yunet_2026may.onnx'))
    parser.add_argument('--model-provenance', type=Path, required=True)
    parser.add_argument('--confidence', type=float, default=0.9)
    parser.add_argument('--margin', type=float, default=0.2, help='Margin per side as a fraction of max face-box dimension (0-2)')
    args = parser.parse_args(argv)
    # FaceDetectorYN lacks an engine argument: select the verified CPU-only engine
    # before importing OpenCV. No implicit ORT/engine fallback is accepted here.
    os.environ['OPENCV_FORCE_DNN_ENGINE'] = '2'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import cv2
    from app.data.face_detection import YuNetDetector
    from app.data.preprocessing import process_manifest
    try:
        detector = YuNetDetector(args.model, args.model_provenance, args.confidence)
        result = process_manifest(args.manifest, args.dataset_root, args.storage_root, detector, args.margin)
    except (OSError, ValueError, KeyError, cv2.error) as exc:
        print(f'Processing stopped: {exc}', file=sys.stderr)
        return 2
    print(f'Status: {result["overall_status"]}\nSummary: {result["summary_path"]}\nReport: {result["report_path"]}')
    return 1 if result['overall_status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
