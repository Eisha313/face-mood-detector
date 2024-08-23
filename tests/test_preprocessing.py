"""Tests for the preprocessing module."""

import pytest
import numpy as np

from face_mood_detector.preprocessing import (
    ImagePreprocessor,
    apply_histogram_equalization,
    apply_clahe,
    augment_image
)


class TestImagePreprocessor:
    """Tests for ImagePreprocessor class."""
    
    def test_init_default_values(self):
        """Test default initialization."""
        preprocessor = ImagePreprocessor()
        
        assert preprocessor.target_size == (48, 48)
        assert preprocessor.normalize is True
        assert preprocessor.grayscale is True
    
    def test_init_custom_values(self):
        """Test custom initialization."""
        preprocessor = ImagePreprocessor(
            target_size=(64, 64),
            normalize=False,
            grayscale=False
        )
        
        assert preprocessor.target_size == (64, 64)
        assert preprocessor.normalize is False
        assert preprocessor.grayscale is False
    
    def test_preprocess_grayscale(self):
        """Test preprocessing with grayscale conversion."""
        preprocessor = ImagePreprocessor(target_size=(48, 48))
        
        # Create a color image
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = preprocessor.preprocess(image)
        
        assert result.shape == (48, 48, 1)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_preprocess_no_normalization(self):
        """Test preprocessing without normalization."""
        preprocessor = ImagePreprocessor(
            target_size=(48, 48),
            normalize=False
        )
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = preprocessor.preprocess(image)
        
        assert result.max() > 1.0 or result.dtype == np.uint8
    
    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        preprocessor = ImagePreprocessor(target_size=(48, 48))
        
        images = [
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        result = preprocessor.preprocess_batch(images)
        
        assert result.shape == (5, 48, 48, 1)
    
    def test_prepare_for_model(self):
        """Test preparation for model input."""
        preprocessor = ImagePreprocessor(target_size=(48, 48))
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = preprocessor.prepare_for_model(image)
        
        assert result.shape == (1, 48, 48, 1)


class TestHistogramEqualization:
    """Tests for histogram equalization functions."""
    
    def test_apply_histogram_equalization_grayscale(self):
        """Test histogram equalization on grayscale image."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        result = apply_histogram_equalization(image)
        
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
    
    def test_apply_histogram_equalization_color(self):
        """Test histogram equalization converts color to grayscale."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = apply_histogram_equalization(image)
        
        assert len(result.shape) == 2
    
    def test_apply_clahe(self):
        """Test CLAHE application."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        result = apply_clahe(image)
        
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
    
    def test_apply_clahe_custom_params(self):
        """Test CLAHE with custom parameters."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        result = apply_clahe(
            image,
            clip_limit=3.0,
            tile_grid_size=(4, 4)
        )
        
        assert result.shape == (100, 100)


class TestAugmentation:
    """Tests for image augmentation."""
    
    def test_augment_flip_horizontal(self):
        """Test horizontal flip augmentation."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:, :50] = 255  # Left half white
        
        result = augment_image(image, flip_horizontal=True)
        
        # After flip, right half should be white
        assert result[:, 50:].mean() > result[:, :50].mean()
    
    def test_augment_rotation(self):
        """Test rotation augmentation."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        result = augment_image(image, rotation_angle=45)
        
        assert result.shape == (100, 100)
    
    def test_augment_brightness(self):
        """Test brightness augmentation."""
        image = np.full((100, 100), 128, dtype=np.uint8)
        
        brighter = augment_image(image, brightness_delta=0.2)
        darker = augment_image(image, brightness_delta=-0.2)
        
        assert brighter.mean() > image.mean()
        assert darker.mean() < image.mean()
    
    def test_augment_no_changes(self):
        """Test augmentation with no changes."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        result = augment_image(image)
        
        np.testing.assert_array_equal(result, image)
