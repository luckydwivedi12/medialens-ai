"""FastAPI application entry point."""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from collections.abc import AsyncIterator
from starlette.types import ASGIApp, Scope, Receive, Send, Message
from starlette.exceptions import HTTPException as StarletteHTTPException

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Must precede any OpenCV import (the pinned YuNet uses native engine 2).
os.environ.setdefault('OPENCV_FORCE_DNN_ENGINE', '2')

from app.api.routes.health import router as health_router
from app.api.routes.detection import router as detection_router
from app.services.images import MAX_UPLOAD_BYTES


def resolve_checkpoint_path() -> Path:
    """Resolve model checkpoint path from environment or local/container defaults."""
    env_path = os.environ.get("MEDIALENS_CHECKPOINT")
    if env_path:
        return Path(env_path)
    for candidate in [
        Path("models/medialens_mobilenetv3/best_model.pth"),
        Path("/app/models/medialens_mobilenetv3/best_model.pth"),
        Path(r"D:\MediaLensData\models\medialens_mobilenetv3\best_model.pth"),
    ]:
        if candidate.is_file():
            return candidate
    return Path("models/medialens_mobilenetv3/best_model.pth")


def resolve_yunet_directory() -> Path:
    """Resolve YuNet detector directory from environment or local/container defaults."""
    env_dir = os.environ.get("MEDIALENS_YUNET_DIRECTORY")
    if env_dir:
        return Path(env_dir)
    for candidate in [
        Path("models/yunet"),
        Path("/app/models/yunet"),
        Path(r"D:\MediaLensData\models\yunet"),
    ]:
        if candidate.is_dir():
            return candidate
    return Path("models/yunet")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    path = resolve_checkpoint_path()
    try:
        from app.core.model_fetcher import ensure_checkpoint_artifact
        path = ensure_checkpoint_artifact(path)
    except Exception:
        logging.getLogger(__name__).exception("Remote checkpoint download failed")

    application.state.inference = None
    application.state.checkpoint_available = path.is_file()
    try:
        from app.ml.inference import ImageInferenceService
        if application.state.checkpoint_available:
            application.state.inference = ImageInferenceService(path)
    except Exception:
        logging.getLogger(__name__).exception("Model startup failed; inference is unavailable")

    application.state.video = None
    try:
        from app.data.face_detection import YuNetDetector, MODEL_VERSION
        from app.services.videos import VideoAnalysisService
        from app.core.model_fetcher import ensure_yunet_artifacts
        directory = resolve_yunet_directory()
        model_path, provenance = ensure_yunet_artifacts(directory, MODEL_VERSION)
        if model_path.is_file() and provenance.is_file():
            application.state.video = VideoAnalysisService(YuNetDetector(model_path, provenance))
    except Exception:
        logging.getLogger(__name__).exception("Video detector startup failed")

    yield
    application.state.video = None
    application.state.inference = None


class InferenceConcurrencyMiddleware:
    """Limit concurrent heavy inference requests to protect memory and CPU on cloud instances."""

    def __init__(self, app: ASGIApp, max_concurrency: int = 2) -> None:
        self.app = app
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") not in {"/api/v1/detect/image", "/api/v1/detect/video"}:
            await self.app(scope, receive, send)
            return

        try:
            # Wait up to 5 seconds to acquire an inference execution slot
            await asyncio.wait_for(self.semaphore.acquire(), timeout=5.0)
        except (asyncio.TimeoutError, TimeoutError):
            await JSONResponse(
                {
                    "success": False,
                    "error": {
                        "code": "server_busy",
                        "message": "The inference server is currently busy processing other media. Please retry shortly.",
                    },
                },
                status_code=503,
            )(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        finally:
            self.semaphore.release()


class UploadLimitMiddleware:
    """Bound multipart requests, including chunked bodies, before disk spooling."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http' or scope.get('method') != 'POST':
            await self.app(scope, receive, send)
            return
        # Spool bounded requests to temporary storage instead of retaining video bytes in RAM.
        from tempfile import SpooledTemporaryFile
        limit = (50 * 1024 * 1024 if scope.get('path') == '/api/v1/detect/video' else MAX_UPLOAD_BYTES) + 65536
        with SpooledTemporaryFile(max_size=1024 * 1024) as body:
            total = 0
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    return
                chunk = message.get('body', b'')
                total += len(chunk)
                if total > limit:
                    await JSONResponse({'success': False, 'error': {'code': 'upload_too_large',
                        'message': 'Upload exceeds the allowed media size.'}}, status_code=413)(scope, receive, send)
                    return
                body.write(chunk)
                if not message.get('more_body', False):
                    break
            body.seek(0)
            delivered = False

            async def replay() -> Message:
                nonlocal delivered
                if delivered:
                    return await receive()
                chunk = body.read(1024 * 1024)
                more = body.tell() < total
                delivered = not more
                return {'type': 'http.request', 'body': chunk, 'more_body': more}

            await self.app(scope, replay, send)


app = FastAPI(title="MediaLens AI", version="0.12.0", lifespan=lifespan)

# Concurrency and size protection middlewares
max_concurrency = int(os.environ.get("MEDIALENS_MAX_CONCURRENT_INFERENCE", "2"))
app.add_middleware(InferenceConcurrencyMiddleware, max_concurrency=max_concurrency)
app.add_middleware(UploadLimitMiddleware)

# Cross-Origin Resource Sharing (CORS) configuration
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
    "https://medialens-ai.vercel.app",
]
extra_origins = os.environ.get("MEDIALENS_ALLOWED_ORIGINS", "").strip()
if extra_origins:
    for origin in extra_origins.split(","):
        clean = origin.strip()
        if clean and clean not in allowed_origins:
            allowed_origins.append(clean)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(health_router)  # Direct /health for cloud providers (Render, ECS, etc.)
app.include_router(detection_router, prefix="/api")


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse({'success': False, 'error': {'code': 'invalid_request',
        'message': 'Send one media file using the multipart field named file.'}}, status_code=422)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse({'success': False, 'error': {'code': str(exc.status_code),
        'message': str(exc.detail)}}, status_code=exc.status_code)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("app.main:app", host=host, port=port)
