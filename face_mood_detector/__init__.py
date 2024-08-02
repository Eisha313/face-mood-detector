"""Face Mood Detector - Real-time facial emotion recognition library."""

__version__ = "0.1.0"
__author__ = "Face Mood Detector Contributors"

from face_mood_detector.detector import EmotionDetector
from face_mood_detector.emotions import Emotion, EmotionResult

__all__ = [
    "EmotionDetector",
    "Emotion",
    "EmotionResult",
    "__version__",
]
