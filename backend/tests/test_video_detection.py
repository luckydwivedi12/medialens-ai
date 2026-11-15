"""Small generated AVI fixtures; no dataset or checkpoint required."""
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.data.face_detection import FaceDetection
from app.services.videos import VideoAnalysisService, inspect_video, video_indices, verify_container, VideoValidationError

class Detector:
    def __init__(self, faces: int = 1) -> None:
        self.faces = faces
        self.calls = 0
    def detect(self, image: Any) -> list[FaceDetection]:
        self.calls += 1
        return [FaceDetection([10., 10., 20., 20.], .99, []) for _ in range(self.faces)]

class Inference:
    threshold = .5
    version = 'unit-fixture'
    def predict(self, image: Any) -> dict[str, Any]:
        return {'probabilities': {'fake': .8, 'real': .2}, 'predicted_label': 'fake'}

@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / 'small.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 64))
    assert writer.isOpened()
    try:
        for i in range(30):
            writer.write(np.full((64, 64, 3), i * 5, dtype=np.uint8))
    finally:
        writer.release()
    return path

@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setitem(app.state._state, 'inference', Inference())
    monkeypatch.setitem(app.state._state, 'video', VideoAnalysisService(Detector()))
    connection = TestClient(app)
    try:
        yield connection
    finally:
        connection.close()



def test_sampling() -> None:
    assert len(video_indices(120)) == 12
    assert video_indices(2) == [0, 1]
    assert len(video_indices(100, 24)) == 24
    assert len(set(video_indices(30))) == 12
    for count, target in [(0, 12), (10, 25)]:
        with pytest.raises(ValueError): video_indices(count, target)


def test_valid_upload_schema_and_cleanup(client: TestClient, video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = app.state.video
    original = service.analyze
    seen: list[Path] = []
    def track(path: Path, *args: Any) -> Any:
        seen.append(path)
        assert path.exists()
        return original(path, *args)
    monkeypatch.setattr(service, 'analyze', track)
    response = client.post('/api/v1/detect/video', files={'file': ('../../private.avi', video.read_bytes(), 'video/x-msvideo')})
    assert response.status_code == 200, response.text
    p = response.json()
    assert p['status'] == 'analyzed' and p['predicted_label'] == 'fake'
    assert p['sampled_frames'] == p['successfully_analyzed_frames'] == p['faces_analyzed'] == 12
    assert p['average_fake_score'] == pytest.approx(.8)
    assert p['frames_above_threshold'] == 12 and p['percentage_frames_above_threshold'] == 100
    assert len(p['frame_level_results']) == 12 and 'not proof' in p['warning']
    assert p['frame_level_results'][0]['faces'][0]['crop_box'] == [6, 6, 34, 34]
    assert all(not path.exists() and not path.parent.exists() for path in seen)
    assert 'private' not in response.text and str(video.parent) not in response.text

@pytest.mark.parametrize(('name', 'content', 'mime', 'status'), [
    ('a.avi', b'junk', 'video/x-msvideo', 400), ('a.avi', b'', 'video/x-msvideo', 400),
    ('a.webm', b'junk', 'video/webm', 415), ('a.mp4', b'RIFFxxxxAVI broken', 'video/mp4', 400),
    ('a.avi', b'RIFFxxxxAVI broken', 'video/x-msvideo', 400),
    ('a.avi', b'x' * (50 * 1024 * 1024 + 1), 'video/x-msvideo', 413),
    ('a.avi', b'x' * (50 * 1024 * 1024 + 70000), 'video/x-msvideo', 413),
])
def test_invalid_uploads(client: TestClient, name: str, content: bytes, mime: str, status: int) -> None:
    response = client.post('/api/v1/detect/video', files={'file': (name, content, mime)})
    assert response.status_code == status and not response.json()['success']


def test_no_faces_and_insufficient(video: Path) -> None:
    service = VideoAnalysisService(Detector(0))
    result = service.analyze(video, '.avi', Inference())
    assert result.status == 'insufficient_analysis' and result.predicted_label is None
    assert result.frames_without_faces == 12 and result.average_fake_score is None
    class TwoFrames(Detector):
        def detect(self, image: Any) -> Any:
            return super().detect(image) if self.calls < 2 else []
    result = VideoAnalysisService(TwoFrames()).analyze(video, '.avi', Inference())
    assert result.successfully_analyzed_frames == 2 and result.predicted_label is None
    assert result.average_fake_score == pytest.approx(.8)


def test_multiple_faces_equal_frame_aggregation(video: Path) -> None:
    class VariableFaces(Detector):
        def detect(self, image: Any) -> Any:
            self.faces = 2 if self.calls == 0 else 1
            return super().detect(image)
    class Scores(Inference):
        calls = 0
        def predict(self, image: Any) -> Any:
            self.calls += 1
            score = .9 if self.calls <= 2 else .1
            return {'probabilities': {'fake': score, 'real': 1-score}, 'predicted_label': 'fake' if score >= .5 else 'real'}
    result = VideoAnalysisService(VariableFaces()).analyze(video, '.avi', Scores())
    assert result.faces_analyzed == 13
    assert result.average_fake_score == pytest.approx((.9 + 11*.1)/12)
    assert result.frames_above_threshold == 1


def test_duration_and_release(monkeypatch: pytest.MonkeyPatch, video: Path) -> None:
    released: list[bool] = []
    class LongCapture:
        def isOpened(self) -> bool: return True
        def get(self, prop: int) -> float:
            return {cv2.CAP_PROP_FRAME_COUNT: 1210, cv2.CAP_PROP_FPS: 10,
                    cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 64}[prop]
        def release(self) -> None: released.append(True)
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *args: LongCapture())
    with pytest.raises(VideoValidationError, match='120 second'): inspect_video(video)
    assert released == [True]


def test_failure_cleanup_and_unavailable(client: TestClient, video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths: list[Path] = []
    def fail(path: Path, *args: Any) -> Any:
        paths.append(path)
        raise RuntimeError(r'D:\secret\weights')
    monkeypatch.setattr(app.state.video, 'analyze', fail)
    response = client.post('/api/v1/detect/video', files={'file': ('a.avi', video.read_bytes(), 'video/x-msvideo')})
    assert response.status_code == 500 and 'secret' not in response.text
    assert not paths[0].parent.exists()
    monkeypatch.setitem(app.state._state, 'inference', None)
    assert client.post('/api/v1/detect/video', files={'file': ('a.avi', video.read_bytes(), 'video/x-msvideo')}).status_code == 503


def test_container_signatures(tmp_path: Path) -> None:
    path = tmp_path / 'header'
    for suffix, header in [('.mp4', b'\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isommp42'),
                           ('.mov', b'\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00qt  '),
                           ('.mkv', b'\x1aE\xdf\xa3\x42\x82\x88matroska')]:
        path.write_bytes(header)
        verify_container(path, suffix)
    path.write_bytes(b'\x1aE\xdf\xa3webm')
    with pytest.raises(ValueError): verify_container(path, '.mkv')


@pytest.mark.parametrize(('suffix', 'codec', 'mime'), [('.mp4', 'mp4v', 'video/mp4'), ('.mov', 'mp4v', 'video/quicktime'), ('.mkv', 'MJPG', 'video/x-matroska')])
def test_other_supported_containers(client: TestClient, tmp_path: Path, suffix: str, codec: str, mime: str) -> None:
    path = tmp_path / ('tiny' + suffix)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 10, (64, 64))
    assert writer.isOpened()
    try:
        for _ in range(5): writer.write(np.zeros((64, 64, 3), np.uint8))
    finally: writer.release()
    response = client.post('/api/v1/detect/video', files={'file': (path.name, path.read_bytes(), mime)})
    assert response.status_code == 200, response.text
    assert response.json()['successfully_analyzed_frames'] == 5


def test_invalid_crop_and_decode_failure_are_recorded(video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.data.frame_extraction import FrameSample
    import app.services.videos as videos
    class InvalidBox(Detector):
        def detect(self, image: Any) -> Any:
            return [FaceDetection([1000., 1000., 20., 20.], .99, [])]
    result = VideoAnalysisService(InvalidBox()).analyze(video, '.avi', Inference())
    assert result.status == 'insufficient_analysis' and result.faces_analyzed == 0
    assert result.frames_without_faces == 0
    assert all(r.status == 'invalid_crops' and r.faces[0].status == 'invalid_crop' for r in result.frame_level_results)
    closed: list[bool] = []
    def samples(*args: Any) -> Any:
        try:
            yield FrameSample(0, 0, error='private internal decoder details')
            for i in range(1, 4): yield FrameSample(i, i/10, np.zeros((64, 64, 3), np.uint8))
        finally: closed.append(True)
    monkeypatch.setattr(videos, 'sample_video', samples)
    result = VideoAnalysisService(Detector()).analyze(video, '.avi', Inference())
    assert result.decoding_failures == 1 and result.successfully_analyzed_frames == 3
    assert result.frame_level_results[0].error == 'Sampled frame could not be decoded.' and closed == [True]
