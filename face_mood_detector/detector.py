"""Main emotion detector class."""

from typing import List, Optional, Union
import numpy as np

from face_mood_detector.emotions import Emotion, EmotionResult


class EmotionDetector:
    """Real-time facial emotion detector using CNN.
    
    This class provides the main interface for detecting emotions
    in images and video streams.
    
    Args:
        model_path: Path to pre-trained model weights. If None, uses default.
        use_gpu: Whether to use GPU acceleration if available.
        temporal_smoothing: Enable smoothing for video stream stability.
        smoothing_window: Number of frames for temporal smoothing.
    
    Example:
        >>> detector = EmotionDetector()
        >>> result = detector.detect_emotion(image)
        >>> print(result.dominant_emotion)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = True,
        temporal_smoothing: bool = True,
        smoothing_window: int = 5,
    ):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_window = smoothing_window
        
        self._model = None
        self._face_detector = None
        self._history: List[np.ndarray] = []
        
    def load_model(self) -> None:
        """Load the CNN model and face detector.
        
        Called automatically on first detection if not called manually.
        """
        # TODO: Implement model loading
        pass
    
    def detect_emotion(
        self,
        image: np.ndarray,
        return_all_faces: bool = False,
    ) -> Union[Optional[EmotionResult], List[EmotionResult]]:
        """Detect emotions in a single image.
        
        Args:
            image: Input image as numpy array (BGR or RGB format).
            return_all_faces: If True, return results for all detected faces.
        
        Returns:
            EmotionResult for the dominant face, list of results if
            return_all_faces is True, or None if no face detected.
        """
        if self._model is None:
            self.load_model()
        
        # TODO: Implement detection pipeline
        # 1. Detect faces
        # 2. Preprocess face regions
        # 3. Run emotion classification
        # 4. Apply temporal smoothing if enabled
        # 5. Return results
        
        return None
    
    def detect_emotions_batch(
        self,
        images: List[np.ndarray],
    ) -> List[Optional[EmotionResult]]:
        """Detect emotions in a batch of images.
        
        Args:
            images: List of input images.
        
        Returns:
            List of EmotionResult objects (or None for images with no faces).
        """
        return [self.detect_emotion(img) for img in images]
    
    def _apply_temporal_smoothing(
        self,
        current_scores: np.ndarray,
    ) -> np.ndarray:
        """Apply temporal smoothing to emotion scores.
        
        Uses a moving average over recent predictions for stability.
        """
        self._history.append(current_scores)
        
        if len(self._history) > self.smoothing_window:
            self._history.pop(0)
        
        if len(self._history) == 0:
            return current_scores
        
        return np.mean(self._history, axis=0)
    
    def reset_smoothing(self) -> None:
        """Reset temporal smoothing history.
        
        Call this when switching between different video streams.
        """
        self._history.clear()
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
