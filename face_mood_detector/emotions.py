"""Emotion classification types and result containers.

This module defines the core emotion types and result structures used
throughout the face mood detector library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Emotion(Enum):
    """Enumeration of the 7 basic emotions supported by the detector.

    These emotions are based on Paul Ekman's research on universal
    facial expressions of emotion.

    Attributes:
        ANGRY: Expression of anger or frustration
        DISGUST: Expression of disgust or distaste
        FEAR: Expression of fear or anxiety
        HAPPY: Expression of happiness or joy
        SAD: Expression of sadness or sorrow
        SURPRISE: Expression of surprise or astonishment
        NEUTRAL: Neutral or baseline expression
    """

    ANGRY = "angry"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"

    @classmethod
    def from_string(cls, value: str) -> Emotion:
        """Create an Emotion from a string value.

        Args:
            value: String representation of the emotion (case-insensitive)

        Returns:
            The corresponding Emotion enum member

        Raises:
            ValueError: If the string doesn't match any emotion

        Example:
            >>> Emotion.from_string("happy")
            <Emotion.HAPPY: 'happy'>
        """
        value_lower = value.lower().strip()
        for emotion in cls:
            if emotion.value == value_lower:
                return emotion
        valid_emotions = [e.value for e in cls]
        raise ValueError(
            f"Unknown emotion: '{value}'. Valid emotions: {valid_emotions}"
        )

    @classmethod
    def list_all(cls) -> List[str]:
        """Get a list of all emotion string values.

        Returns:
            List of emotion names as strings

        Example:
            >>> Emotion.list_all()
            ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        """
        return [emotion.value for emotion in cls]


@dataclass
class EmotionResult:
    """Container for emotion detection results.

    Holds the detection results including the dominant emotion,
    confidence scores for all emotions, and optional face location.

    Attributes:
        dominant_emotion: The emotion with highest confidence
        confidence: Confidence score for the dominant emotion (0.0 to 1.0)
        all_scores: Dictionary mapping each emotion to its confidence score
        face_location: Optional tuple of (x, y, width, height) for detected face
        timestamp: Optional timestamp for video frame results

    Example:
        >>> result = EmotionResult(
        ...     dominant_emotion=Emotion.HAPPY,
        ...     confidence=0.95,
        ...     all_scores={Emotion.HAPPY: 0.95, Emotion.NEUTRAL: 0.05}
        ... )
        >>> result.is_confident(threshold=0.8)
        True
    """

    dominant_emotion: Emotion
    confidence: float
    all_scores: Dict[Emotion, float] = field(default_factory=dict)
    face_location: Optional[tuple[int, int, int, int]] = None
    timestamp: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate the result after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

    def is_confident(self, threshold: float = 0.5) -> bool:
        """Check if the detection confidence meets a threshold.

        Args:
            threshold: Minimum confidence level (default: 0.5)

        Returns:
            True if confidence >= threshold, False otherwise
        """
        return self.confidence >= threshold

    def to_dict(self) -> Dict[str, any]:
        """Convert the result to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "dominant_emotion": self.dominant_emotion.value,
            "confidence": round(self.confidence, 4),
            "all_scores": {
                emotion.value: round(score, 4)
                for emotion, score in self.all_scores.items()
            },
            "face_location": self.face_location,
            "timestamp": self.timestamp,
        }

    def get_top_emotions(self, n: int = 3) -> List[tuple[Emotion, float]]:
        """Get the top N emotions by confidence score.

        Args:
            n: Number of top emotions to return (default: 3)

        Returns:
            List of (Emotion, confidence) tuples sorted by confidence descending
        """
        sorted_scores = sorted(
            self.all_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_scores[:n]

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return (
            f"EmotionResult({self.dominant_emotion.value}: "
            f"{self.confidence:.1%} confidence)"
        )

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        return (
            f"EmotionResult(dominant_emotion={self.dominant_emotion!r}, "
            f"confidence={self.confidence:.4f}, "
            f"all_scores={{...}}, "
            f"face_location={self.face_location})"
        )
