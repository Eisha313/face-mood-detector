"""Face detection and preprocessing module.

Provides face detection capabilities using Haar cascades and
preprocessing utilities for emotion recognition.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass

from .config import DetectorConfig


class BoundingBox(NamedTuple):
    """Represents a face bounding box."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        """Get area of bounding box."""
        return self.width * self.height
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, w, h) tuple."""
        return (self.x, self.y, self.width, self.height)


@dataclass
class DetectedFace:
    """Container for detected face data."""
    bounding_box: BoundingBox
    face_image: np.ndarray
    preprocessed: np.ndarray
    confidence: float = 1.0
    
    @property
    def location(self) -> Tuple[int, int, int, int]:
        """Get face location as tuple."""
        return self.bounding_box.to_tuple()


class FaceDetector:
    """Handles face detection and preprocessing for emotion recognition."""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize face detector.
        
        Args:
            config: Detector configuration. Uses defaults if not provided.
        """
        self.config = config or DetectorConfig()
        self._cascade = None
        self._load_cascade()
    
    def _load_cascade(self) -> None:
        """Load Haar cascade classifier."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._cascade = cv2.CascadeClassifier(cascade_path)
        
        if self._cascade.empty():
            raise RuntimeError("Failed to load Haar cascade classifier")
    
    def detect_faces(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect all faces in an image.
        
        Args:
            image: Input image (BGR or grayscale).
            
        Returns:
            List of DetectedFace objects.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect faces
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=(self.config.min_face_size, self.config.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detected_faces = []
        for (x, y, w, h) in faces:
            bbox = BoundingBox(x, y, w, h)
            face_image = gray[y:y+h, x:x+w]
            preprocessed = self._preprocess_face(face_image)
            
            detected_faces.append(DetectedFace(
                bounding_box=bbox,
                face_image=face_image,
                preprocessed=preprocessed
            ))
        
        # Sort by face area (largest first)
        detected_faces.sort(key=lambda f: f.bounding_box.area, reverse=True)
        
        return detected_faces
    
    def detect_largest_face(self, image: np.ndarray) -> Optional[DetectedFace]:
        """Detect the largest face in an image.
        
        Args:
            image: Input image (BGR or grayscale).
            
        Returns:
            DetectedFace object or None if no face found.
        """
        faces = self.detect_faces(image)
        return faces[0] if faces else None
    
    def _preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for emotion recognition model.
        
        Args:
            face_image: Cropped grayscale face image.
            
        Returns:
            Preprocessed image ready for model input.
        """
        # Resize to model input size
        input_size = self.config.input_size
        resized = cv2.resize(face_image, (input_size, input_size))
        
        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(resized)
        
        # Normalize to [0, 1]
        normalized = equalized.astype(np.float32) / 255.0
        
        # Add channel dimension for CNN (H, W) -> (H, W, 1)
        preprocessed = np.expand_dims(normalized, axis=-1)
        
        return preprocessed
    
    def preprocess_batch(self, faces: List[DetectedFace]) -> np.ndarray:
        """Create a batch from multiple detected faces.
        
        Args:
            faces: List of DetectedFace objects.
            
        Returns:
            Batch array of shape (N, H, W, 1).
        """
        if not faces:
            return np.array([])
        
        batch = np.stack([f.preprocessed for f in faces], axis=0)
        return batch
    
    def draw_detections(
        self,
        image: np.ndarray,
        faces: List[DetectedFace],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw bounding boxes on image.
        
        Args:
            image: Input image to draw on.
            faces: List of detected faces.
            color: Box color in BGR format.
            thickness: Line thickness.
            
        Returns:
            Image with drawn bounding boxes.
        """
        output = image.copy()
        
        for face in faces:
            x, y, w, h = face.bounding_box.to_tuple()
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
        
        return output
