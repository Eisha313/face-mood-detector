"""Face Mood Detector - Real-time facial emotion recognition library."""

from .emotions import Emotion, EmotionResult, EMOTION_LABELS
from .detector import FaceDetector
from .config import Config, ModelConfig, TrainingConfig, DetectionConfig, default_config

__version__ = "0.1.0"
__author__ = "Face Mood Detector Contributors"

__all__ = [
    "Emotion",
    "EmotionResult",
    "EMOTION_LABELS",
    "FaceDetector",
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "DetectionConfig",
    "default_config",
]
