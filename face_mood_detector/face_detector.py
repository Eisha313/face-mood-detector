"""Face detection module using Haar cascades and DNN."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detects faces in images using OpenCV."""
    
    def __init__(self, method: str = "haar", min_confidence: float = 0.5):
        """
        Initialize face detector.
        
        Args:
            method: Detection method ('haar' or 'dnn')
            min_confidence: Minimum confidence threshold for DNN method
        """
        self.method = method
        self.min_confidence = min_confidence
        self._cascade = None
        self._net = None
        self._initialized = False
        
        self._initialize_detector()
    
    def _initialize_detector(self) -> None:
        """Initialize the selected detection method."""
        if self.method == "haar":
            self._initialize_haar()
        elif self.method == "dnn":
            self._initialize_dnn()
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
    
    def _initialize_haar(self) -> None:
        """Initialize Haar cascade classifier."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        
        if not Path(cascade_path).exists():
            logger.error(f"Haar cascade file not found: {cascade_path}")
            raise FileNotFoundError(f"Haar cascade not found: {cascade_path}")
        
        self._cascade = cv2.CascadeClassifier(cascade_path)
        
        if self._cascade.empty():
            logger.error("Failed to load Haar cascade classifier")
            raise RuntimeError("Failed to load Haar cascade")
        
        self._initialized = True
        logger.info("Haar cascade face detector initialized")
    
    def _initialize_dnn(self) -> None:
        """Initialize DNN-based face detector."""
        # For now, fall back to Haar if DNN models not available
        logger.warning("DNN method not fully implemented, falling back to Haar")
        self.method = "haar"
        self._initialize_haar()
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            List of face bounding boxes as (x, y, width, height)
        """
        if not self._initialized:
            logger.error("Face detector not initialized")
            return []
        
        if image is None:
            logger.warning("Received None image for face detection")
            return []
        
        if image.size == 0:
            logger.warning("Received empty image for face detection")
            return []
        
        # Validate image dimensions
        if len(image.shape) < 2:
            logger.warning("Invalid image dimensions")
            return []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                logger.warning(f"Unexpected number of channels: {image.shape[2]}")
                return []
        else:
            gray = image.copy()
        
        # Ensure proper dtype
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        
        if self.method == "haar":
            return self._detect_haar(gray)
        else:
            return self._detect_dnn(image)
    
    def _detect_haar(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using Haar cascade.
        
        Args:
            gray: Grayscale image
            
        Returns:
            List of face bounding boxes
        """
        try:
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                return []
            
            # Validate and convert detections
            valid_faces = []
            img_height, img_width = gray.shape[:2]
            
            for (x, y, w, h) in faces:
                # Ensure coordinates are within image bounds
                x = max(0, int(x))
                y = max(0, int(y))
                w = min(int(w), img_width - x)
                h = min(int(h), img_height - y)
                
                # Skip invalid detections
                if w <= 0 or h <= 0:
                    continue
                
                valid_faces.append((x, y, w, h))
            
            return valid_faces
            
        except cv2.error as e:
            logger.error(f"OpenCV error during face detection: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during face detection: {e}")
            return []
    
    def _detect_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using DNN method.
        
        Args:
            image: Input BGR image
            
        Returns:
            List of face bounding boxes
        """
        # Placeholder for DNN implementation
        logger.warning("DNN detection not implemented")
        return []
    
    def extract_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                     target_size: Tuple[int, int] = (48, 48)) -> Optional[np.ndarray]:
        """
        Extract and preprocess a face region from an image.
        
        Args:
            image: Input image
            bbox: Face bounding box (x, y, w, h)
            target_size: Output size for the face image
            
        Returns:
            Preprocessed face image or None if extraction fails
        """
        if image is None or image.size == 0:
            logger.warning("Invalid image for face extraction")
            return None
        
        x, y, w, h = bbox
        
        # Validate bounding box
        if w <= 0 or h <= 0:
            logger.warning(f"Invalid bounding box dimensions: w={w}, h={h}")
            return None
        
        img_height, img_width = image.shape[:2]
        
        # Clamp coordinates to image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_width, x + w)
        y2 = min(img_height, y + h)
        
        # Check if we have a valid region
        if x2 <= x1 or y2 <= y1:
            logger.warning("Bounding box outside image bounds")
            return None
        
        try:
            # Extract face region
            face_region = image[y1:y2, x1:x2]
            
            if face_region.size == 0:
                logger.warning("Extracted empty face region")
                return None
            
            # Convert to grayscale if needed
            if len(face_region.shape) == 3:
                face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_region
            
            # Resize to target size
            face_resized = cv2.resize(
                face_gray, 
                target_size,
                interpolation=cv2.INTER_AREA
            )
            
            return face_resized
            
        except cv2.error as e:
            logger.error(f"OpenCV error during face extraction: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during face extraction: {e}")
            return None
