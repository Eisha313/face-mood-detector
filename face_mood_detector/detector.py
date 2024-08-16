"""Main emotion detector module."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

from .emotions import Emotion, EmotionResult
from .face_detector import FaceDetector
from .config import DetectorConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class EmotionDetector:
    """Main class for detecting emotions in faces."""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize the emotion detector.
        
        Args:
            config: Configuration options for the detector
        """
        self.config = config or DEFAULT_CONFIG
        self._face_detector = None
        self._model = None
        self._initialized = False
        self._emotion_history: List[Dict[Emotion, float]] = []
        
    def initialize(self) -> bool:
        """
        Initialize the detector and load models.
        
        Returns:
            True if initialization successful
        """
        try:
            self._face_detector = FaceDetector(
                method=self.config.face_detection_method,
                min_confidence=self.config.min_face_confidence
            )
            
            # Model loading will be implemented later
            self._model = None
            self._initialized = True
            
            logger.info("EmotionDetector initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize EmotionDetector: {e}")
            self._initialized = False
            return False
    
    @property
    def is_initialized(self) -> bool:
        """Check if detector is initialized."""
        return self._initialized
    
    def detect_emotions(self, image: np.ndarray) -> List[EmotionResult]:
        """
        Detect emotions in all faces in an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of EmotionResult for each detected face
        """
        if not self._initialized:
            logger.warning("Detector not initialized. Call initialize() first.")
            if not self.initialize():
                return []
        
        # Validate input image
        if image is None:
            logger.warning("Received None image")
            return []
        
        if not isinstance(image, np.ndarray):
            logger.warning(f"Expected numpy array, got {type(image)}")
            return []
        
        if image.size == 0:
            logger.warning("Received empty image")
            return []
        
        if len(image.shape) < 2:
            logger.warning(f"Invalid image shape: {image.shape}")
            return []
        
        # Detect faces
        try:
            faces = self._face_detector.detect_faces(image)
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
        
        if not faces:
            logger.debug("No faces detected in image")
            return []
        
        results = []
        
        for bbox in faces:
            try:
                # Extract face region
                face = self._face_detector.extract_face(
                    image, 
                    bbox,
                    target_size=(self.config.input_size, self.config.input_size)
                )
                
                if face is None:
                    logger.debug(f"Failed to extract face at {bbox}")
                    continue
                
                # Predict emotions
                emotion_scores = self._predict_emotions(face)
                
                if emotion_scores is None:
                    logger.debug(f"Failed to predict emotions for face at {bbox}")
                    continue
                
                # Apply temporal smoothing if enabled
                if self.config.enable_temporal_smoothing:
                    emotion_scores = self._apply_temporal_smoothing(emotion_scores)
                
                # Get dominant emotion
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[dominant_emotion]
                
                result = EmotionResult(
                    emotion=dominant_emotion,
                    confidence=confidence,
                    all_scores=emotion_scores,
                    face_bbox=bbox
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing face at {bbox}: {e}")
                continue
        
        return results
    
    def _predict_emotions(self, face: np.ndarray) -> Optional[Dict[Emotion, float]]:
        """
        Predict emotions for a preprocessed face image.
        
        Args:
            face: Preprocessed face image
            
        Returns:
            Dictionary mapping emotions to confidence scores
        """
        if face is None or face.size == 0:
            return None
        
        # Placeholder: return dummy predictions
        # Real implementation will use the CNN model
        try:
            # Generate placeholder scores
            scores = {
                Emotion.HAPPY: 0.15,
                Emotion.SAD: 0.10,
                Emotion.ANGRY: 0.10,
                Emotion.FEAR: 0.10,
                Emotion.SURPRISE: 0.15,
                Emotion.DISGUST: 0.10,
                Emotion.NEUTRAL: 0.30
            }
            
            # Normalize scores to sum to 1.0
            total = sum(scores.values())
            if total > 0:
                scores = {k: v / total for k, v in scores.items()}
            else:
                # Fallback to uniform distribution
                uniform_score = 1.0 / len(Emotion)
                scores = {e: uniform_score for e in Emotion}
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in emotion prediction: {e}")
            return None
    
    def _apply_temporal_smoothing(self, scores: Dict[Emotion, float]) -> Dict[Emotion, float]:
        """
        Apply temporal smoothing to emotion scores.
        
        Args:
            scores: Current emotion scores
            
        Returns:
            Smoothed emotion scores
        """
        if not scores:
            return scores
        
        self._emotion_history.append(scores)
        
        # Keep only recent history
        window = self.config.smoothing_window
        if len(self._emotion_history) > window:
            self._emotion_history = self._emotion_history[-window:]
        
        if len(self._emotion_history) < 2:
            return scores
        
        # Average scores over history
        smoothed = {}
        for emotion in Emotion:
            values = [h.get(emotion, 0.0) for h in self._emotion_history]
            smoothed[emotion] = sum(values) / len(values)
        
        # Renormalize
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}
        
        return smoothed
    
    def reset_history(self) -> None:
        """Reset temporal smoothing history."""
        self._emotion_history.clear()
        logger.debug("Emotion history cleared")
    
    def detect_single_face(self, image: np.ndarray) -> Optional[EmotionResult]:
        """
        Detect emotion for the largest face in an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            EmotionResult for the largest face, or None if no face found
        """
        results = self.detect_emotions(image)
        
        if not results:
            return None
        
        # Return result for largest face (by area)
        largest = max(
            results,
            key=lambda r: r.face_bbox[2] * r.face_bbox[3] if r.face_bbox else 0
        )
        
        return largest
