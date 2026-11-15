"""Tests for the public health contract and local CORS policy."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "medialens-ai"}


def test_direct_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "medialens-ai"}


def test_local_frontend_cors() -> None:
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_vercel_frontend_cors() -> None:
    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://medialens-ai.vercel.app",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://medialens-ai.vercel.app"


def test_unapproved_origin_has_no_cors_permission() -> None:
    response = client.get("/api/health", headers={"Origin": "https://example.com"})

    assert "access-control-allow-origin" not in response.headers
