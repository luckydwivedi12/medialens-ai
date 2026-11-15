"""Service health endpoint."""

from fastapi import APIRouter

from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return service availability without invoking inference."""
    return HealthResponse(status="ok", service="medialens-ai")
