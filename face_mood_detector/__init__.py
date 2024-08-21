"""Face Mood Detector - Real-time facial emotion recognition."""

from .emotions import Emotion, EMOTION_LABELS, get_emotion_color
from .config import Config, get_default_config, load_config
from .face_detector import FaceDetector, DetectedFace
from .detector import EmotionDetector
from .smoother import EmotionSmoother, MultiTrackSmoother

__version__ = "0.1.0"
__all__ = [
    "Emotion",
    "EMOTION_LABELS",
    "get_emotion_color",
    "Config",
    "get_default_config",
    "load_config",
    "FaceDetector",
    "DetectedFace",
    "EmotionDetector",
    "EmotionSmoother",
    "MultiTrackSmoother",
]
