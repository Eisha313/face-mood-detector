"""Tests for emotion smoother module."""

import pytest

from face_mood_detector.emotions import Emotion
from face_mood_detector.smoother import EmotionSmoother, MultiTrackSmoother


class TestEmotionSmoother:
    """Tests for EmotionSmoother class."""
    
    def test_initialization(self):
        """Test smoother initialization with default parameters."""
        smoother = EmotionSmoother()
        assert smoother.window_size == 5
        assert smoother.decay_factor == 0.8
        assert smoother.min_confidence_threshold == 0.3
    
    def test_invalid_window_size(self):
        """Test that invalid window size raises error."""
        with pytest.raises(ValueError):
            EmotionSmoother(window_size=0)
    
    def test_invalid_decay_factor(self):
        """Test that invalid decay factor raises error."""
        with pytest.raises(ValueError):
            EmotionSmoother(decay_factor=1.5)
        with pytest.raises(ValueError):
            EmotionSmoother(decay_factor=0)
    
    def test_single_update(self):
        """Test smoothing with single prediction."""
        smoother = EmotionSmoother(window_size=3)
        scores = {Emotion.HAPPY: 0.8, Emotion.NEUTRAL: 0.2}
        
        smoothed = smoother.update(scores)
        
        assert Emotion.HAPPY in smoothed
        assert smoothed[Emotion.HAPPY] > smoothed[Emotion.NEUTRAL]
    
    def test_multiple_updates_smoothing(self):
        """Test that multiple updates produce smoothed results."""
        smoother = EmotionSmoother(window_size=3, decay_factor=0.5)
        
        # First prediction: happy
        smoother.update({Emotion.HAPPY: 0.9, Emotion.SAD: 0.1})
        
        # Second prediction: sad
        smoother.update({Emotion.HAPPY: 0.1, Emotion.SAD: 0.9})
        
        # Third prediction: sad
        smoothed = smoother.update({Emotion.HAPPY: 0.1, Emotion.SAD: 0.9})
        
        # Sad should be dominant but happy should still have some weight
        assert smoothed[Emotion.SAD] > smoothed[Emotion.HAPPY]
    
    def test_get_dominant_emotion(self):
        """Test getting dominant emotion."""
        smoother = EmotionSmoother()
        
        # No history
        assert smoother.get_dominant_emotion() is None
        
        # After update
        smoother.update({Emotion.ANGRY: 0.7, Emotion.NEUTRAL: 0.3})
        assert smoother.get_dominant_emotion() == Emotion.ANGRY
    
    def test_reset(self):
        """Test resetting smoother history."""
        smoother = EmotionSmoother()
        smoother.update({Emotion.HAPPY: 0.9})
        
        smoother.reset()
        
        assert smoother.get_dominant_emotion() is None
    
    def test_is_stable(self):
        """Test stability detection."""
        smoother = EmotionSmoother(window_size=4)
        
        # Not enough history
        smoother.update({Emotion.HAPPY: 0.9})
        assert not smoother.is_stable
        
        # Consistent predictions
        smoother.update({Emotion.HAPPY: 0.9})
        smoother.update({Emotion.HAPPY: 0.85})
        smoother.update({Emotion.HAPPY: 0.95})
        assert smoother.is_stable
    
    def test_confidence_property(self):
        """Test confidence score calculation."""
        smoother = EmotionSmoother()
        
        # No history
        assert smoother.confidence == 0.0
        
        # After update
        smoother.update({Emotion.FEAR: 0.8, Emotion.SURPRISE: 0.2})
        assert smoother.confidence > 0.0


class TestMultiTrackSmoother:
    """Tests for MultiTrackSmoother class."""
    
    def test_multiple_tracks(self):
        """Test handling multiple face tracks."""
        multi_smoother = MultiTrackSmoother(window_size=3)
        
        # Update track 1
        result1 = multi_smoother.update(1, {Emotion.HAPPY: 0.9})
        
        # Update track 2
        result2 = multi_smoother.update(2, {Emotion.SAD: 0.8})
        
        assert result1[Emotion.HAPPY] > 0.5
        assert result2[Emotion.SAD] > 0.5
    
    def test_track_cleanup(self):
        """Test that old tracks are cleaned up."""
        multi_smoother = MultiTrackSmoother()
        multi_smoother._max_age = 5
        
        multi_smoother.update(1, {Emotion.HAPPY: 0.9})
        
        # Advance frames without updating track 1
        for _ in range(10):
            multi_smoother.step()
        
        assert multi_smoother.get_smoother(1) is None
    
    def test_reset(self):
        """Test resetting all smoothers."""
        multi_smoother = MultiTrackSmoother()
        multi_smoother.update(1, {Emotion.HAPPY: 0.9})
        multi_smoother.update(2, {Emotion.SAD: 0.8})
        
        multi_smoother.reset()
        
        assert multi_smoother.get_smoother(1) is None
        assert multi_smoother.get_smoother(2) is None
