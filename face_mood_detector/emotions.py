"""Emotion definitions and result containers."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
import time


class Emotion(Enum):
    """Seven basic emotions for facial expression classification."""
    
    ANGRY = "angry"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"
    
    @classmethod
    def from_index(cls, index: int) -> "Emotion":
        """Get emotion from model output index."""
        mapping = [
            cls.ANGRY,
            cls.DISGUST,
            cls.FEAR,
            cls.HAPPY,
            cls.SAD,
            cls.SURPRISE,
            cls.NEUTRAL,
        ]
        if 0 <= index < len(mapping):
            return mapping[index]
        raise ValueError(f"Invalid emotion index: {index}")
    
    @classmethod
    def count(cls) -> int:
        """Return the number of emotion classes."""
        return len(cls)


@dataclass
class EmotionResult:
    """Container for emotion detection results."""
    
    dominant_emotion: Emotion
    confidence: float
    all_scores: Dict[Emotion, float]
    face_bbox: Optional[tuple] = None  # (x, y, width, height)
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    @property
    def confidence_percent(self) -> float:
        """Return confidence as percentage."""
        return self.confidence * 100
    
    def get_top_emotions(self, n: int = 3) -> list:
        """Get top N emotions by confidence score."""
        sorted_emotions = sorted(
            self.all_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_emotions[:n]
    
    def to_dict(self) -> dict:
        """Convert result to dictionary for serialization."""
        return {
            "dominant_emotion": self.dominant_emotion.value,
            "confidence": self.confidence,
            "confidence_percent": self.confidence_percent,
            "all_scores": {e.value: s for e, s in self.all_scores.items()},
            "face_bbox": self.face_bbox,
            "timestamp": self.timestamp,
        }
