"""Face Mood Detector - Real-time facial emotion recognition.

A lightweight Python library for detecting emotions from facial expressions
using CNN-based deep learning.

Example:
    >>> from face_mood_detector import EmotionDetector
    >>> detector = EmotionDetector()
    >>> result = detector.detect_emotion(image)
    >>> print(result.dominant_emotion)
"""

from .emotions import Emotion, EmotionResult, EMOTION_LABELS
from .detector import EmotionDetector
from .config import DetectorConfig, ModelConfig
from .face_detector import FaceDetector, DetectedFace, BoundingBox

__version__ = "0.1.0"
__author__ = "Face Mood Detector Contributors"

__all__ = [
    # Main classes
    "EmotionDetector",
    "FaceDetector",
    
    # Data classes
    "Emotion",
    "EmotionResult",
    "DetectedFace",
    "BoundingBox",
    
    # Configuration
    "DetectorConfig",
    "ModelConfig",
    
    # Constants
    "EMOTION_LABELS",
]
