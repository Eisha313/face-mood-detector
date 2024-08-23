"""Main emotion detector module using CNN-based classification."""

import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from .emotions import Emotion, EmotionResult
from .config import ModelConfig, get_default_config
from .face_detector import FaceDetector
from .smoother import EmotionSmoother
from .preprocessing import ImagePreprocessor, apply_clahe


class EmotionDetector:
    """CNN-based facial emotion detector."""
    
    # Emotion labels in order of model output
    EMOTION_LABELS = [
        Emotion.ANGRY,
        Emotion.DISGUST,
        Emotion.FEAR,
        Emotion.HAPPY,
        Emotion.SAD,
        Emotion.SURPRISE,
        Emotion.NEUTRAL
    ]
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_path: Optional[str] = None,
        use_smoothing: bool = True,
        apply_contrast_enhancement: bool = True
    ):
        """
        Initialize the emotion detector.
        
        Args:
            config: Model configuration. Uses default if not provided.
            model_path: Path to pre-trained model weights.
            use_smoothing: Whether to apply temporal smoothing.
            apply_contrast_enhancement: Whether to apply CLAHE preprocessing.
        """
        self.config = config or get_default_config()
        self.model_path = model_path
        self.model = None
        self.face_detector = FaceDetector()
        self.preprocessor = ImagePreprocessor(
            target_size=self.config.input_size,
            normalize=True,
            grayscale=True
        )
        self.apply_contrast_enhancement = apply_contrast_enhancement
        
        self.smoother = EmotionSmoother() if use_smoothing else None
        self._is_loaded = False
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the pre-trained CNN model.
        
        Args:
            model_path: Path to model weights. Uses default if not provided.
        """
        path = model_path or self.model_path
        
        # Placeholder for actual model loading
        # In production, this would load a TensorFlow/PyTorch model
        self._is_loaded = True
        self.model = self._create_mock_model()
    
    def _create_mock_model(self):
        """Create a mock model for testing purposes."""
        class MockModel:
            def predict(self, x):
                batch_size = x.shape[0]
                # Return random probabilities that sum to 1
                probs = np.random.rand(batch_size, 7)
                return probs / probs.sum(axis=1, keepdims=True)
        
        return MockModel()
    
    def _preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess a face image for model input.
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Preprocessed image ready for model
        """
        # Apply contrast enhancement if enabled
        if self.apply_contrast_enhancement:
            face_image = apply_clahe(face_image)
        
        return self.preprocessor.prepare_for_model(face_image)
    
    def detect_emotion(self, image: np.ndarray) -> Optional[EmotionResult]:
        """
        Detect emotion from a single face image.
        
        Args:
            image: Face image as numpy array (BGR format)
            
        Returns:
            EmotionResult with detected emotion and confidence scores,
            or None if detection fails.
        """
        if not self._is_loaded:
            self.load_model()
        
        # Preprocess the image
        processed = self._preprocess_face(image)
        
        # Get model predictions
        predictions = self.model.predict(processed)[0]
        
        # Create confidence dictionary
        confidences = {
            emotion: float(predictions[i])
            for i, emotion in enumerate(self.EMOTION_LABELS)
        }
        
        # Find dominant emotion
        dominant_idx = np.argmax(predictions)
        dominant_emotion = self.EMOTION_LABELS[dominant_idx]
        confidence = float(predictions[dominant_idx])
        
        result = EmotionResult(
            emotion=dominant_emotion,
            confidence=confidence,
            all_confidences=confidences
        )
        
        # Apply smoothing if enabled
        if self.smoother is not None:
            result = self.smoother.smooth(result)
        
        return result
    
    def detect_emotions_in_frame(
        self,
        frame: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], EmotionResult]]:
        """
        Detect faces and emotions in a full frame.
        
        Args:
            frame: Full image frame (BGR format)
            
        Returns:
            List of tuples containing (bounding_box, emotion_result)
        """
        if not self._is_loaded:
            self.load_model()
        
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Detect emotion
            emotion_result = self.detect_emotion(face_roi)
            
            if emotion_result is not None:
                results.append(((x, y, w, h), emotion_result))
        
        return results
    
    def reset_smoothing(self) -> None:
        """Reset the temporal smoothing state."""
        if self.smoother is not None:
            self.smoother.reset()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
