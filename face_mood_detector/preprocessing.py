"""Image preprocessing utilities for emotion detection."""

import numpy as np
from typing import Tuple, Optional
import cv2


class ImagePreprocessor:
    """Handles image preprocessing for the emotion detection model."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (48, 48),
        normalize: bool = True,
        grayscale: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target image dimensions (height, width)
            normalize: Whether to normalize pixel values to [0, 1]
            grayscale: Whether to convert images to grayscale
        """
        self.target_size = target_size
        self.normalize = normalize
        self.grayscale = grayscale
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image for model input.
        
        Args:
            image: Input image as numpy array (BGR or grayscale)
            
        Returns:
            Preprocessed image ready for model input
        """
        processed = image.copy()
        
        # Convert to grayscale if needed
        if self.grayscale and len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        processed = cv2.resize(
            processed,
            (self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_AREA
        )
        
        # Normalize pixel values
        if self.normalize:
            processed = processed.astype(np.float32) / 255.0
        
        # Add channel dimension if grayscale
        if self.grayscale and len(processed.shape) == 2:
            processed = np.expand_dims(processed, axis=-1)
        
        return processed
    
    def preprocess_batch(
        self,
        images: list,
        add_batch_dim: bool = True
    ) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of images as numpy arrays
            add_batch_dim: Whether to stack into batch array
            
        Returns:
            Batch of preprocessed images
        """
        processed = [self.preprocess(img) for img in images]
        
        if add_batch_dim:
            return np.stack(processed, axis=0)
        return processed
    
    def prepare_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        Fully prepare an image for model inference.
        
        Args:
            image: Input image
            
        Returns:
            Image with batch dimension added, ready for inference
        """
        processed = self.preprocess(image)
        return np.expand_dims(processed, axis=0)


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to improve contrast.
    
    Args:
        image: Grayscale image
        
    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(image)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    return clahe.apply(image)


def augment_image(
    image: np.ndarray,
    flip_horizontal: bool = False,
    rotation_angle: Optional[float] = None,
    brightness_delta: Optional[float] = None
) -> np.ndarray:
    """
    Apply augmentation to an image.
    
    Args:
        image: Input image
        flip_horizontal: Whether to flip horizontally
        rotation_angle: Rotation angle in degrees
        brightness_delta: Brightness adjustment (-1 to 1)
        
    Returns:
        Augmented image
    """
    result = image.copy()
    
    if flip_horizontal:
        result = cv2.flip(result, 1)
    
    if rotation_angle is not None:
        h, w = result.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        result = cv2.warpAffine(result, matrix, (w, h))
    
    if brightness_delta is not None:
        result = result.astype(np.float32)
        result = result + (brightness_delta * 255)
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result
