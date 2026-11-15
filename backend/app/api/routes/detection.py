"""Bounded, ephemeral image prediction API."""

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.schemas.detection import ModelStatus, PredictionResponse
from app.schemas.video import VideoResponse
from app.services.images import MAX_UPLOAD_BYTES, MIME_FORMATS, decode_image

router = APIRouter(prefix="/v1", tags=["inference"])
logger = logging.getLogger(__name__)


@router.get("/model/status", response_model=ModelStatus)
def model_status(request: Request) -> ModelStatus:
    service = getattr(request.app.state, "inference", None)
    return ModelStatus(
        video_ready=service is not None and getattr(request.app.state, 'video', None) is not None,
        model_ready=service is not None,
        checkpoint_available=getattr(request.app.state, "checkpoint_available", False),
        model_version=service.version if service else None,
        threshold=service.threshold if service else None,
        error=None if service else "Model unavailable. Check backend configuration and restart the service.",
    )


@router.post("/detect/image", response_model=PredictionResponse)
async def detect_image(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    try:
        if file.content_type not in MIME_FORMATS:
            raise HTTPException(415, detail="Use a JPEG, PNG or WEBP image.")
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, detail="Image exceeds the 10 MB upload limit.")
        try:
            image = await run_in_threadpool(decode_image, content, file.content_type)
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc
        try:
            service = getattr(request.app.state, "inference", None)
            if service is None:
                raise HTTPException(503, detail="Model unavailable. Check model status and restart the backend.")
            try:
                return PredictionResponse(**await run_in_threadpool(service.predict, image))
            except Exception as exc:
                logger.exception("Image inference failed")
                raise HTTPException(500, detail="Image analysis failed. Please retry with a different image.") from exc
        finally:
            image.close()
    finally:
        await file.close()


@router.post('/detect/video', response_model=VideoResponse)
async def detect_video(request: Request, file: UploadFile = File(...)) -> VideoResponse:
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from app.services.videos import MAX_VIDEO_BYTES, VIDEO_TYPES, VideoValidationError

    try:
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in VIDEO_TYPES or file.content_type not in VIDEO_TYPES[suffix] | {'application/octet-stream'}:
            raise HTTPException(415, detail='Use an MP4, MOV, AVI or MKV video with a matching media type.')
        inference = getattr(request.app.state, 'inference', None)
        video = getattr(request.app.state, 'video', None)
        if inference is None or video is None:
            raise HTTPException(503, detail='Video analysis unavailable. Check model and detector configuration, then restart.')
        with TemporaryDirectory(prefix='medialens-video-') as directory:
            path = Path(directory) / ('upload' + suffix)
            size = 0
            with path.open('xb') as destination:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_VIDEO_BYTES:
                        raise HTTPException(413, detail='Video exceeds the 50 MB upload limit.')
                    destination.write(chunk)
            if not size:
                raise HTTPException(400, detail='Choose a nonempty video.')
            try:
                return await run_in_threadpool(video.analyze, path, suffix, inference)
            except VideoValidationError as exc:
                raise HTTPException(400, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception('Video analysis failed')
                raise HTTPException(500, detail='Video analysis failed. Please retry with another video.') from exc
    finally:
        await file.close()
