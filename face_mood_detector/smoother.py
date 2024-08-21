"""Temporal smoothing for stable emotion predictions."""

from collections import deque
from typing import Dict, List, Optional
import numpy as np

from .emotions import Emotion


class EmotionSmoother:
    """Applies temporal smoothing to emotion predictions for stability.
    
    Uses a sliding window approach with configurable weighting to reduce
    jitter and provide more stable emotion readings over time.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        decay_factor: float = 0.8,
        min_confidence_threshold: float = 0.3
    ):
        """Initialize the emotion smoother.
        
        Args:
            window_size: Number of frames to consider for smoothing.
            decay_factor: Exponential decay factor for older predictions (0-1).
                Higher values give more weight to recent predictions.
            min_confidence_threshold: Minimum confidence to consider a prediction.
        """
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if not 0 < decay_factor <= 1:
            raise ValueError("decay_factor must be between 0 and 1")
        if not 0 <= min_confidence_threshold <= 1:
            raise ValueError("min_confidence_threshold must be between 0 and 1")
        
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.min_confidence_threshold = min_confidence_threshold
        
        # Store history of predictions
        self._history: deque = deque(maxlen=window_size)
        self._weights: np.ndarray = self._compute_weights()
    
    def _compute_weights(self) -> np.ndarray:
        """Compute exponential decay weights for the window."""
        weights = np.array([
            self.decay_factor ** i 
            for i in range(self.window_size - 1, -1, -1)
        ])
        return weights / weights.sum()
    
    def update(self, scores: Dict[Emotion, float]) -> Dict[Emotion, float]:
        """Add new prediction and return smoothed scores.
        
        Args:
            scores: Dictionary mapping emotions to confidence scores.
            
        Returns:
            Smoothed emotion scores.
        """
        # Filter out low confidence predictions
        filtered_scores = {
            emotion: score 
            for emotion, score in scores.items()
            if score >= self.min_confidence_threshold
        }
        
        # If all scores are below threshold, use original scores
        if not filtered_scores:
            filtered_scores = scores
        
        self._history.append(filtered_scores)
        return self._compute_smoothed_scores()
    
    def _compute_smoothed_scores(self) -> Dict[Emotion, float]:
        """Compute weighted average of scores in history."""
        if not self._history:
            return {emotion: 0.0 for emotion in Emotion}
        
        # Get weights for current history length
        current_weights = self._weights[-len(self._history):]
        current_weights = current_weights / current_weights.sum()
        
        # Compute weighted average for each emotion
        smoothed = {}
        for emotion in Emotion:
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for i, scores in enumerate(self._history):
                if emotion in scores:
                    weighted_sum += scores[emotion] * current_weights[i]
                    weight_sum += current_weights[i]
            
            if weight_sum > 0:
                smoothed[emotion] = weighted_sum / weight_sum
            else:
                smoothed[emotion] = 0.0
        
        # Normalize to sum to 1
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}
        
        return smoothed
    
    def get_dominant_emotion(self) -> Optional[Emotion]:
        """Get the current dominant emotion from smoothed predictions.
        
        Returns:
            The emotion with highest smoothed confidence, or None if no history.
        """
        if not self._history:
            return None
        
        smoothed = self._compute_smoothed_scores()
        return max(smoothed, key=smoothed.get)
    
    def reset(self) -> None:
        """Clear the prediction history."""
        self._history.clear()
    
    @property
    def is_stable(self) -> bool:
        """Check if predictions have stabilized.
        
        Returns True if the same emotion has been dominant for
        at least half the window size.
        """
        if len(self._history) < self.window_size // 2:
            return False
        
        recent_dominant = []
        for scores in list(self._history)[-self.window_size // 2:]:
            if scores:
                dominant = max(scores, key=scores.get)
                recent_dominant.append(dominant)
        
        if not recent_dominant:
            return False
        
        # Check if all recent predictions agree
        return len(set(recent_dominant)) == 1
    
    @property
    def confidence(self) -> float:
        """Get confidence level of current smoothed prediction.
        
        Returns:
            Confidence score between 0 and 1.
        """
        if not self._history:
            return 0.0
        
        smoothed = self._compute_smoothed_scores()
        if not smoothed:
            return 0.0
        
        return max(smoothed.values())


class MultiTrackSmoother:
    """Manages emotion smoothing for multiple tracked faces."""
    
    def __init__(self, **smoother_kwargs):
        """Initialize multi-track smoother.
        
        Args:
            **smoother_kwargs: Arguments passed to EmotionSmoother instances.
        """
        self._smoother_kwargs = smoother_kwargs
        self._smoothers: Dict[int, EmotionSmoother] = {}
        self._last_seen: Dict[int, int] = {}
        self._frame_count = 0
        self._max_age = 30  # Remove tracks not seen for this many frames
    
    def update(
        self,
        track_id: int,
        scores: Dict[Emotion, float]
    ) -> Dict[Emotion, float]:
        """Update smoother for a specific track.
        
        Args:
            track_id: Unique identifier for the tracked face.
            scores: Emotion scores for this track.
            
        Returns:
            Smoothed emotion scores.
        """
        if track_id not in self._smoothers:
            self._smoothers[track_id] = EmotionSmoother(**self._smoother_kwargs)
        
        self._last_seen[track_id] = self._frame_count
        return self._smoothers[track_id].update(scores)
    
    def step(self) -> None:
        """Advance frame counter and cleanup old tracks."""
        self._frame_count += 1
        self._cleanup_old_tracks()
    
    def _cleanup_old_tracks(self) -> None:
        """Remove tracks that haven't been seen recently."""
        to_remove = [
            track_id 
            for track_id, last_seen in self._last_seen.items()
            if self._frame_count - last_seen > self._max_age
        ]
        
        for track_id in to_remove:
            del self._smoothers[track_id]
            del self._last_seen[track_id]
    
    def get_smoother(self, track_id: int) -> Optional[EmotionSmoother]:
        """Get smoother for a specific track."""
        return self._smoothers.get(track_id)
    
    def reset(self) -> None:
        """Reset all smoothers."""
        self._smoothers.clear()
        self._last_seen.clear()
        self._frame_count = 0
