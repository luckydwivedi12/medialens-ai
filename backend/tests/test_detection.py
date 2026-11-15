"""API edge cases use a controlled unit fixture; real checkpoint smoke is separate."""

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.ml.dataset import build_transform, preprocessing_config
from app.ml.inference import ImageInferenceService
from app.services.images import MAX_UPLOAD_BYTES, decode_image


def image_bytes(fmt: str = 'PNG', mode: str = 'RGB') -> bytes:
    output = BytesIO()
    Image.new(mode, (30, 40), 120).save(output, format=fmt)
    return output.getvalue()


@pytest.fixture
def service() -> ImageInferenceService:
    # Bypass disk loading only in this unit fixture. Production never uses fixtures.
    instance = ImageInferenceService.__new__(ImageInferenceService)
    class FixedLogits(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            assert not torch.is_grad_enabled() and tensor.shape == (1, 3, 224, 224)
            return torch.tensor([[0.2, 1.1]])
    instance.model = FixedLogits().eval()
    instance.class_to_index = {'fake': 1, 'real': 0}
    instance.transform = build_transform(False, preprocessing_config())
    instance.threshold = .5
    instance.version = 'unit-test-fixture'
    instance.lock = Lock()
    return instance


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, service: ImageInferenceService) -> Iterator[TestClient]:
    monkeypatch.setitem(app.state._state, 'inference', service)
    monkeypatch.setitem(app.state._state, 'checkpoint_available', True)
    connection = TestClient(app)
    try:
        yield connection
    finally:
        connection.close()


@pytest.mark.parametrize(('fmt', 'mime'), [('JPEG', 'image/jpeg'), ('PNG', 'image/png'), ('WEBP', 'image/webp')])
def test_valid_image_and_prediction_schema(client: TestClient, fmt: str, mime: str) -> None:
    response = client.post('/api/v1/detect/image', files={'file': ('../../private.jpg', image_bytes(fmt), mime)})
    assert response.status_code == 200
    payload = response.json()
    expected = torch.softmax(torch.tensor([.2, 1.1]), dim=0)
    assert payload['probabilities']['real'] == pytest.approx(float(expected[0]))
    assert payload['probabilities']['fake'] == pytest.approx(float(expected[1]))
    assert payload['predicted_label'] == 'fake' and payload['threshold'] == .5
    assert payload['model_name'] == 'MobileNetV3-Large' and payload['device'] == 'cpu'
    assert 'not proof' in payload['warning']
    assert 'path' not in response.text and 'private' not in response.text


@pytest.mark.parametrize(('content', 'mime', 'status'), [
    (b'not an image', 'image/png', 400), (b'', 'image/png', 400),
    (image_bytes()[:45], 'image/png', 400), (image_bytes(), 'text/plain', 415),
    (image_bytes(), 'image/jpeg', 400), (image_bytes('GIF'), 'image/png', 400),
    (b'x' * (MAX_UPLOAD_BYTES + 1), 'image/png', 413),
    (b'x' * (MAX_UPLOAD_BYTES + 70000), 'image/png', 413),
])
def test_rejected_uploads(client: TestClient, content: bytes, mime: str, status: int) -> None:
    response = client.post('/api/v1/detect/image', files={'file': ('file.png', content, mime)})
    assert response.status_code == status
    assert response.json()['success'] is False
    assert response.json()['error']['message']


def test_missing_file_and_model_status(client: TestClient) -> None:
    assert client.post('/api/v1/detect/image').status_code == 422
    status = client.get('/api/v1/model/status').json()
    assert status['model_ready'] and status['checkpoint_available']
    assert status['device'] == 'cpu' and status['model_version'] == 'unit-test-fixture'
    assert client.get('/api/health').json()['status'] == 'ok'


def test_unavailable_model(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(app.state._state, 'inference', None)
    status = client.get('/api/v1/model/status').json()
    assert status['model_ready'] is False and status['model_version'] is None
    response = client.post('/api/v1/detect/image', files={'file': ('a.png', image_bytes(), 'image/png')})
    assert response.status_code == 503


def test_inference_error_is_sanitized(client: TestClient, service: ImageInferenceService, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(image: Image.Image) -> dict[str, Any]:
        raise RuntimeError(r'D:\secret\checkpoint.pth')
    monkeypatch.setattr(service, 'predict', fail)
    response = client.post('/api/v1/detect/image', files={'file': ('a.png', image_bytes(), 'image/png')})
    assert response.status_code == 500 and 'secret' not in response.text


def test_startup_load_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('MEDIALENS_CHECKPOINT', str(tmp_path / 'absent.pth'))
    with TestClient(app) as client:
        assert client.get('/api/health').status_code == 200
        status = client.get('/api/v1/model/status').json()
        assert not status['model_ready'] and not status['checkpoint_available']
        assert str(tmp_path) not in str(status)


def test_exif_orientation_and_rgb() -> None:
    output = BytesIO()
    image = Image.new('RGB', (30, 40))
    exif = image.getexif(); exif[274] = 6
    image.save(output, format='JPEG', exif=exif)
    with decode_image(output.getvalue(), 'image/jpeg') as decoded:
        assert decoded.size == (40, 30) and decoded.mode == 'RGB'
    with decode_image(image_bytes(mode='L'), 'image/png') as decoded:
        assert decoded.mode == 'RGB'


def test_animation_and_pixel_limit_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    output = BytesIO()
    Image.new('RGB', (30, 40)).save(output, format='PNG', save_all=True,
                                   append_images=[Image.new('RGB', (30, 40), 'red')], duration=100)
    with pytest.raises(ValueError, match='animated'):
        decode_image(output.getvalue(), 'image/png')
    monkeypatch.setattr('app.services.images.MAX_PIXELS', 100)
    with pytest.raises(ValueError, match='megapixel'):
        decode_image(image_bytes(), 'image/png')


def test_post_cors(client: TestClient) -> None:
    response = client.options('/api/v1/detect/image', headers={'Origin': 'http://localhost:5173',
        'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type'})
    assert response.status_code == 200
    assert response.headers['access-control-allow-origin'] == 'http://localhost:5173'


def test_chunked_body_is_bounded(client: TestClient) -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(12):
            yield b'x' * 1024 * 1024
    response = client.post('/api/v1/detect/image', content=chunks(), headers={'Content-Type': 'image/png'})
    assert response.status_code == 413


def test_threshold_boundary(service: ImageInferenceService) -> None:
    with Image.new('RGB', (30, 40)) as image:
        service.threshold = service.predict(image)['probabilities']['fake']
        assert service.predict(image)['predicted_label'] == 'fake'
        service.threshold += .001
        assert service.predict(image)['predicted_label'] == 'real'
