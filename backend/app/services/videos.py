"""Ephemeral video validation and sequential face inference. No dataset labels/masks."""
import math
from pathlib import Path
from threading import Lock
from time import perf_counter

import cv2
from PIL import Image
from app.data.face_cropping import crop_face
from app.data.face_detection import YuNetDetector
from app.data.frame_extraction import sample_video
from app.ml.inference import ImageInferenceService
from app.schemas.detection import Probabilities
from app.schemas.video import FaceResult, FrameResult, VideoResponse

class VideoValidationError(ValueError):
    """Safe public validation message."""


MAX_VIDEO_BYTES = 50 * 1024 * 1024
MAX_DURATION = 120
VIDEO_TYPES = {'.mp4': {'video/mp4'}, '.mov': {'video/quicktime'},
               '.avi': {'video/x-msvideo', 'video/avi'}, '.mkv': {'video/x-matroska'}}


def video_indices(count: int, target: int = 12) -> list[int]:
    if count < 1 or not 1 <= target <= 24:
        raise ValueError('Invalid sampling configuration')
    return sorted({min(count - 1, int(count * (i + .5) / target)) for i in range(target)})


def verify_container(path: Path, suffix: str) -> None:
    with path.open('rb') as handle:
        header = handle.read(4096)
    valid = False
    if suffix == '.avi':
        valid = header[:4] == b'RIFF' and header[8:12] == b'AVI '
    elif suffix == '.mkv':
        # Require Matroska DocType, not merely the shared WebM EBML magic.
        valid = header[:4] == b'\x1aE\xdf\xa3' and b'\x42\x82\x88matroska' in header
    elif suffix in {'.mp4', '.mov'}:
        if header[4:8] == b'ftyp':
            size = int.from_bytes(header[:4], 'big')
            brands = [header[i:i+4] for i in range(8, min(size, len(header)), 4) if i != 12]
            allowed = {b'qt  '} if suffix == '.mov' else {b'isom', b'iso2', b'iso4', b'iso5', b'iso6', b'mp41', b'mp42', b'avc1', b'M4V '}
            valid = size >= 16 and bool(allowed.intersection(brands))
        elif suffix == '.mov':
            valid = header[4:8] in {b'moov', b'mdat', b'wide'}
    if not valid:
        raise VideoValidationError('Video container does not match the supported file format.')


def inspect_video(path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoValidationError('Video is corrupt or its codec is unsupported.')
        count, fps = capture.get(cv2.CAP_PROP_FRAME_COUNT), capture.get(cv2.CAP_PROP_FPS)
        width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not all(math.isfinite(v) and v > 0 for v in (count, fps, width, height)):
            raise VideoValidationError('Video has invalid duration or dimensions.')
        if width * height > 3840 * 2160 or fps > 120:
            raise VideoValidationError('Video exceeds the 4K or 120 FPS resource limit.')
        if count / fps > MAX_DURATION:
            raise VideoValidationError('Video exceeds the 120 second duration limit.')
        # Sequential decode validation catches broken frames outside the sample, bounded by duration.
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            if frame is None or frame.size == 0 or frame.shape[:2] != (int(height), int(width)):
                raise VideoValidationError('Video contains corrupt frames.')
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if decoded / fps > MAX_DURATION or timestamp > MAX_DURATION:
                raise VideoValidationError('Video exceeds the 120 second duration limit.')
        if decoded != int(count):
            raise VideoValidationError('Video is truncated or has inconsistent frame metadata.')
        return decoded, max(count / fps, timestamp if decoded else 0)
    finally:
        capture.release()


class VideoAnalysisService:
    def __init__(self, detector: YuNetDetector) -> None:
        self.detector = detector
        self.lock = Lock()

    def analyze(self, path: Path, suffix: str, inference: ImageInferenceService) -> VideoResponse:
        started = perf_counter()
        results: list[FrameResult] = []
        # YuNet has mutable input dimensions; serialize complete video jobs on this CPU.
        with self.lock:
            verify_container(path, suffix)
            count, duration = inspect_video(path)
            samples = sample_video(path, video_indices(count))
            try:
                for sample in samples:
                    if sample.frame_index is None:
                        raise VideoValidationError('Video could not be decoded.')
                    row = FrameResult(frame_index=sample.frame_index,
                        timestamp_seconds=sample.timestamp_seconds or 0,
                        timestamp_method=sample.timestamp_method, status='decode_failed')
                    if sample.image is None:
                        row.error = 'Sampled frame could not be decoded.'
                    else:
                        detections = self.detector.detect(sample.image)
                        for detection in detections:
                            face = FaceResult(bounding_box=detection.coordinates,
                                detection_confidence=detection.confidence, status='invalid_crop')
                            try:
                                crop, box = crop_face(sample.image, detection.coordinates, margin=.2)
                            except ValueError:
                                row.faces.append(face)
                                continue
                            with Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) as image:
                                prediction = inference.predict(image)
                            face.crop_box = list(box)
                            face.status = 'analyzed'
                            face.probabilities = Probabilities(**prediction['probabilities'])
                            face.predicted_label = prediction['predicted_label']
                            row.faces.append(face)
                        scores = [f.probabilities.fake for f in row.faces if f.status == 'analyzed']
                        row.status = 'analyzed' if scores else 'invalid_crops' if detections else 'no_faces'
                        if scores:
                            row.average_fake_score = sum(scores) / len(scores)
                    results.append(row)
            finally:
                samples.close()
        scores = [r.average_fake_score for r in results if r.average_fake_score is not None]
        average = sum(scores) / len(scores) if scores else None
        real_scores = [sum(f.probabilities.real for f in r.faces if f.status == 'analyzed') /
                       sum(f.status == 'analyzed' for f in r.faces) for r in results if r.status == 'analyzed']
        average_real = sum(real_scores) / len(real_scores) if real_scores else None
        sufficient = len(scores) >= 3
        above = sum(s >= inference.threshold for s in scores)
        return VideoResponse(status='analyzed' if sufficient else 'insufficient_analysis',
            predicted_label=('fake' if average >= inference.threshold else 'real') if sufficient else None,
            average_fake_score=average, average_real_score=average_real,
            threshold=inference.threshold, sampled_frames=len(results), successfully_analyzed_frames=len(scores),
            frames_without_faces=sum(r.status == 'no_faces' for r in results),
            faces_analyzed=sum(f.status == 'analyzed' for r in results for f in r.faces),
            frames_above_threshold=above, percentage_frames_above_threshold=100*above/len(scores) if scores else 0,
            decoding_failures=sum(r.status == 'decode_failed' for r in results), frame_level_results=results,
            duration_seconds=duration, processing_time_seconds=perf_counter()-started, model_version=inference.version,
            limitations=['Experimental facial-crop classifier; video-level performance has not been validated.',
                'Aggregate scores are not calibrated probabilities of the entire video.',
                'Uniform frame-index sampling can miss brief events; variable-frame-rate timestamps may be approximate.',
                'No face tracking, target selection, audio analysis or manipulation masks; bystanders influence the average.',
                'Fewer than three analyzed frames means insufficient analysis; no classification is issued.'])
