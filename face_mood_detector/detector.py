"""Main emotion detector module.

Provides the primary interface for emotion detection from images and video.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from .emotions import Emotion, EmotionResult, EMOTION_LABELS
from .config import DetectorConfig, ModelConfig
from .face_detector import FaceDetector, DetectedFace


class EmotionDetector:
    """Main class for detecting emotions from faces.
    
    This class provides a high-level interface for emotion detection,
    combining face detection with emotion classification.
    
    Example:
        >>> detector = EmotionDetector()
        >>> result = detector.detect_emotion(image)
        >>> print(result.dominant_emotion)
    """
    
    def __init__(
        self,
        detector_config: Optional[DetectorConfig] = None,
        model_config: Optional[ModelConfig] = None,
        model_path: Optional[Union[str, Path]] = None
    ):
        """Initialize the emotion detector.
        
        Args:
            detector_config: Configuration for face detection.
            model_config: Configuration for emotion model.
            model_path: Path to pre-trained model weights.
        """
        self.detector_config = detector_config or DetectorConfig()
        self.model_config = model_config or ModelConfig()
        self.model_path = Path(model_path) if model_path else None
        
        self._face_detector = FaceDetector(self.detector_config)
        self._model = None
        self._model_loaded = False
    
    def _ensure_model_loaded(self) -> None:
        """Ensure the emotion model is loaded."""
        if not self._model_loaded:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the emotion recognition model."""
        # Placeholder for model loading
        # Will be implemented when CNN module is added
        self._model_loaded = True
    
    def detect_emotion(self, image: np.ndarray) -> Optional[EmotionResult]:
        """Detect emotion from the largest face in an image.
        
        Args:
            image: Input image as numpy array (BGR format).
            
        Returns:
            EmotionResult if a face is detected, None otherwise.
        """
        face = self._face_detector.detect_largest_face(image)
        if face is None:
            return None
        
        return self._classify_emotion(face)
    
    def detect_all_emotions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect emotions from all faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format).
            
        Returns:
            List of dicts containing face location and emotion result.
        """
        faces = self._face_detector.detect_faces(image)
        
        results = []
        for face in faces:
            emotion_result = self._classify_emotion(face)
            results.append({
                'location': face.location,
                'emotion': emotion_result
            })
        
        return results
    
    def _classify_emotion(self, face: DetectedFace) -> EmotionResult:
        """Classify emotion from a detected face.
        
        Args:
            face: DetectedFace object with preprocessed image.
            
        Returns:
            EmotionResult with emotion predictions.
        """
        self._ensure_model_loaded()
        
        # Placeholder: generate dummy predictions
        # Will be replaced with actual model inference
        predictions = self._mock_predictions()
        
        return EmotionResult.from_predictions(predictions)
    
    def _mock_predictions(self) -> np.ndarray:
        """Generate mock predictions for testing.
        
        Returns:
            Array of 7 probability values.
        """
        # Generate random probabilities that sum to 1
        raw = np.random.rand(len(EMOTION_LABELS))
        probabilities = raw / raw.sum()
        return probabilities
    
    def get_face_locations(self, image: np.ndarray) -> List[tuple]:
        """Get bounding boxes of all detected faces.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            List of (x, y, width, height) tuples.
        """
        faces = self._face_detector.detect_faces(image)
        return [face.location for face in faces]
    
    def draw_results(
        self,
        image: np.ndarray,
        show_confidence: bool = True
    ) -> np.ndarray:
        """Detect emotions and draw results on image.
        
        Args:
            image: Input image.
            show_confidence: Whether to show confidence scores.
            
        Returns:
            Image with drawn emotion results.
        """
        import cv2
        
        output = image.copy()
        results = self.detect_all_emotions(image)
        
        for result in results:
            x, y, w, h = result['location']
            emotion = result['emotion']
            
            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw emotion label
            label = emotion.dominant_emotion.value
            if show_confidence:
                label += f" ({emotion.confidence:.1%})"
            
            cv2.putText(
                output, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        
        return output
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
