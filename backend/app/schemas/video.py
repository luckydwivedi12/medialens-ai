"""Public video assessment schema; aggregate scores are not calibrated probabilities."""
from typing import Literal
from pydantic import BaseModel, Field
from app.schemas.detection import WARNING, Probabilities

class FaceResult(BaseModel):
    bounding_box: list[float]
    detection_confidence: float
    crop_box: list[int] | None = None
    status: Literal['analyzed', 'invalid_crop']
    probabilities: Probabilities | None = None
    predicted_label: Literal['real', 'fake'] | None = None

class FrameResult(BaseModel):
    frame_index: int
    timestamp_seconds: float
    timestamp_method: str
    status: Literal['analyzed', 'no_faces', 'invalid_crops', 'decode_failed']
    average_fake_score: float | None = Field(default=None, ge=0, le=1)
    faces: list[FaceResult] = Field(default_factory=list)
    error: str | None = None

class VideoResponse(BaseModel):
    success: Literal[True] = True
    status: Literal['analyzed', 'insufficient_analysis']
    predicted_label: Literal['real', 'fake'] | None
    average_fake_score: float | None = Field(ge=0, le=1)
    average_real_score: float | None = Field(ge=0, le=1)
    threshold: float
    sampled_frames: int
    successfully_analyzed_frames: int
    frames_without_faces: int
    faces_analyzed: int
    frames_above_threshold: int
    percentage_frames_above_threshold: float
    decoding_failures: int
    frame_level_results: list[FrameResult]
    duration_seconds: float
    processing_time_seconds: float
    model_name: str = 'MobileNetV3-Large'
    model_version: str
    warning: str = WARNING
    limitations: list[str]
