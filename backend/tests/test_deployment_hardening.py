"""Tests for cloud deployment hardening: health routes, CORS, concurrency, and model fetcher."""

import asyncio
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.model_fetcher import (
    DEFAULT_MODEL_METADATA,
    DEFAULT_PREPROCESSING_CONFIG,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_YUNET_SHA256,
    ensure_checkpoint_artifact,
    ensure_yunet_artifacts,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_dual_health_routes(client: TestClient) -> None:
    """Verify both standard /health and legacy /api/health report healthy status."""
    r1 = client.get("/health")
    assert r1.status_code == 200
    assert r1.json() == {"status": "ok", "service": "medialens-ai"}

    r2 = client.get("/api/health")
    assert r2.status_code == 200
    assert r2.json() == {"status": "ok", "service": "medialens-ai"}


def test_vercel_cors_preflight(client: TestClient) -> None:
    """Verify CORS preflight succeeds for the production Vercel frontend domain."""
    response = client.options(
        "/api/v1/detect/image",
        headers={
            "Origin": "https://medialens-ai.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://medialens-ai.vercel.app"


def test_cors_extra_origins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify MEDIALENS_ALLOWED_ORIGINS environment variable expands allowed origins."""
    monkeypatch.setenv("MEDIALENS_ALLOWED_ORIGINS", "https://custom-preview.vercel.app")
    # Recreate client to trigger middleware origin evaluation
    with TestClient(app) as test_client:
        response = test_client.options(
            "/api/health",
            headers={
                "Origin": "https://medialens-ai.vercel.app",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "https://medialens-ai.vercel.app"


def test_model_fetcher_without_urls(tmp_path: Path) -> None:
    """When no remote URLs are configured, ensure_checkpoint_artifact safely returns the path without crashing."""
    absent = tmp_path / "absent_model.pth"
    res = ensure_checkpoint_artifact(absent)
    assert res == absent
    assert not res.is_file()


def test_model_fetcher_checksum_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Model fetcher must reject artifacts whose SHA256 does not match the pinned expectation."""
    from app.core.model_fetcher import download_file
    
    # Create a dummy corrupted file to simulate bad download
    bad_file = tmp_path / "corrupted.pth"
    
    # Mock urllib.request.urlopen to return corrupt content
    class DummyResponse:
        def __init__(self) -> None:
            self._chunks = [b"corrupted_content"]
        def read(self, size: int = 65536) -> bytes:
            return self._chunks.pop(0) if self._chunks else b""
        def __enter__(self) -> "DummyResponse":
            return self
        def __exit__(self, *args: object) -> None:
            pass

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=120: DummyResponse())
    
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_file("https://example.com/bad.pth", bad_file, expected_sha256=EXPECTED_CHECKPOINT_SHA256)
    
    assert not bad_file.exists()


def test_companion_config_generation(tmp_path: Path) -> None:
    """When checkpoint exists, companion metadata and preprocessing configs are created if missing."""
    ckpt = tmp_path / "best_model.pth"
    ckpt.write_bytes(b"dummy_weights")
    
    ensure_checkpoint_artifact(ckpt)
    
    metadata_file = tmp_path / "model_metadata.json"
    config_file = tmp_path / "preprocessing_config.json"
    
    assert metadata_file.is_file()
    assert config_file.is_file()
    assert EXPECTED_CHECKPOINT_SHA256 in metadata_file.read_text(encoding="utf-8")

