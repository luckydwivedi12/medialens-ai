"""Public inference contracts contain no filesystem paths or checkpoint contents."""

from typing import Literal

from pydantic import BaseModel, Field

WARNING = "Experimental research model. Prediction is not proof of authenticity. Scores are not calibrated probabilities of real-world authenticity."


class Probabilities(BaseModel):
    real: float = Field(ge=0, le=1)
    fake: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    success: Literal[True] = True
    predicted_label: Literal["real", "fake"]
    probabilities: Probabilities
    threshold: float
    model_name: str
    model_version: str
    device: Literal["cpu"] = "cpu"
    warning: str = WARNING


class ModelStatus(BaseModel):
    video_ready: bool = False
    model_ready: bool
    architecture: str = "MobileNetV3-Large"
    device: Literal["cpu"] = "cpu"
    model_version: str | None = None
    checkpoint_available: bool
    threshold: float | None = None
    error: str | None = None
