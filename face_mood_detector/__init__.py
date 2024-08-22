"""Face Mood Detector - Real-time facial emotion recognition."""

from .emotions import Emotion, EmotionResult
from .config import Config, ModelConfig, DetectionConfig
from .face_detector import FaceDetector
from .smoother import EmotionSmoother, SmoothedEmotion
from .detector import EmotionDetector
from .video_analyzer import VideoAnalyzer, FrameResult

__version__ = "0.1.0"
__all__ = [
    "Emotion",
    "EmotionResult",
    "Config",
    "ModelConfig",
    "DetectionConfig",
    "FaceDetector",
    "EmotionSmoother",
    "SmoothedEmotion",
    "EmotionDetector",
    "VideoAnalyzer",
    "FrameResult",
]
