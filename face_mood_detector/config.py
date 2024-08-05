"""Configuration settings for face mood detector."""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any
import os


@dataclass
class ModelConfig:
    """Configuration for the CNN model architecture."""
    
    # Input image dimensions
    input_size: Tuple[int, int] = (48, 48)
    num_channels: int = 1  # Grayscale
    
    # Model architecture
    num_classes: int = 7
    dropout_rate: float = 0.25
    
    # Convolutional layers
    conv_filters: Tuple[int, ...] = (32, 64, 128, 256)
    kernel_size: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 2)
    
    # Dense layers
    dense_units: Tuple[int, ...] = (512, 256)
    
    # Batch normalization
    use_batch_norm: bool = True


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    
    # Learning rate schedule
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 5
    min_learning_rate: float = 1e-7
    
    # Early stopping
    early_stopping_patience: int = 10
    
    # Data augmentation
    use_augmentation: bool = True
    rotation_range: int = 15
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    horizontal_flip: bool = True
    zoom_range: float = 0.1
    
    # Validation
    validation_split: float = 0.2


@dataclass
class DetectionConfig:
    """Configuration for real-time detection."""
    
    # Face detection
    face_cascade_path: str = "haarcascade_frontalface_default.xml"
    min_face_size: Tuple[int, int] = (30, 30)
    scale_factor: float = 1.1
    min_neighbors: int = 5
    
    # Temporal smoothing
    use_smoothing: bool = True
    smoothing_window: int = 5
    smoothing_method: str = "exponential"  # "exponential" or "moving_average"
    exponential_alpha: float = 0.3
    
    # Confidence threshold
    min_confidence: float = 0.3
    
    # Performance
    skip_frames: int = 0  # Process every Nth frame (0 = process all)
    max_faces: int = 10


@dataclass
class Config:
    """Main configuration container."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    
    # Paths
    model_dir: str = "models"
    default_model_name: str = "emotion_model.h5"
    cache_dir: str = ".cache"
    
    # Logging
    log_level: str = "INFO"
    
    @property
    def default_model_path(self) -> str:
        """Get the default model file path."""
        return os.path.join(self.model_dir, self.default_model_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": {
                "input_size": self.model.input_size,
                "num_channels": self.model.num_channels,
                "num_classes": self.model.num_classes,
                "dropout_rate": self.model.dropout_rate,
                "conv_filters": self.model.conv_filters,
                "kernel_size": self.model.kernel_size,
                "pool_size": self.model.pool_size,
                "dense_units": self.model.dense_units,
                "use_batch_norm": self.model.use_batch_norm,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "epochs": self.training.epochs,
                "learning_rate": self.training.learning_rate,
                "lr_decay_factor": self.training.lr_decay_factor,
                "lr_decay_patience": self.training.lr_decay_patience,
                "early_stopping_patience": self.training.early_stopping_patience,
                "use_augmentation": self.training.use_augmentation,
                "validation_split": self.training.validation_split,
            },
            "detection": {
                "min_face_size": self.detection.min_face_size,
                "scale_factor": self.detection.scale_factor,
                "min_neighbors": self.detection.min_neighbors,
                "use_smoothing": self.detection.use_smoothing,
                "smoothing_window": self.detection.smoothing_window,
                "min_confidence": self.detection.min_confidence,
            },
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        config = cls()
        
        if "model" in config_dict:
            for key, value in config_dict["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        if "training" in config_dict:
            for key, value in config_dict["training"].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        if "detection" in config_dict:
            for key, value in config_dict["detection"].items():
                if hasattr(config.detection, key):
                    setattr(config.detection, key, value)
        
        return config


# Default configuration instance
default_config = Config()
