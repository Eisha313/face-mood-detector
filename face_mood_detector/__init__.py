"""Face Mood Detector - Real-time facial emotion recognition library.

A lightweight Python library for real-time facial emotion recognition
using CNN-based deep learning with support for webcam feeds, single images,
and video streams.

Example:
    >>> from face_mood_detector import EmotionDetector
    >>> detector = EmotionDetector()
    >>> result = detector.detect_from_image("photo.jpg")
    >>> print(result.dominant_emotion)
    'happy'

Modules:
    - detector: Main emotion detection interface
    - video_analyzer: Video stream analysis utilities
    - emotions: Emotion classification definitions
    - preprocessing: Image preprocessing utilities
    - smoother: Temporal smoothing for stable predictions
    - face_detector: Face detection utilities
    - config: Configuration management
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from face_mood_detector.emotions import Emotion, EmotionResult
from face_mood_detector.detector import EmotionDetector
from face_mood_detector.video_analyzer import VideoAnalyzer
from face_mood_detector.smoother import EmotionSmoother
from face_mood_detector.config import DetectorConfig
from face_mood_detector.preprocessing import preprocess_face, normalize_image

__version__ = "0.1.0"
__author__ = "Face Mood Detector Contributors"
__license__ = "MIT"

__all__ = [
    # Core classes
    "EmotionDetector",
    "VideoAnalyzer",
    "EmotionSmoother",
    "DetectorConfig",
    # Data types
    "Emotion",
    "EmotionResult",
    # Utilities
    "preprocess_face",
    "normalize_image",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]
