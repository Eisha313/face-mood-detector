"""Configuration management for the face mood detector.

This module provides configuration classes for customizing the behavior
of the emotion detector, including model settings, detection parameters,
and output options.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


# Default paths and constants
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "emotion_model.h5"
DEFAULT_CASCADE_PATH = Path(__file__).parent / "data" / "haarcascade_frontalface_default.xml"

# Environment variable overrides
ENV_MODEL_PATH = "FACE_MOOD_MODEL_PATH"
ENV_CASCADE_PATH = "FACE_MOOD_CASCADE_PATH"
ENV_USE_GPU = "FACE_MOOD_USE_GPU"


@dataclass
class DetectorConfig:
    """Configuration options for the EmotionDetector.

    This class holds all configurable parameters for emotion detection,
    including model paths, detection thresholds, and processing options.

    Attributes:
        model_path: Path to the trained emotion recognition model
        cascade_path: Path to the Haar cascade for face detection
        use_gpu: Whether to use GPU acceleration if available
        confidence_threshold: Minimum confidence for valid detections
        face_min_size: Minimum face size in pixels (width, height)
        face_scale_factor: Scale factor for face detection pyramid
        face_min_neighbors: Minimum neighbors for face detection
        input_size: Expected input size for the emotion model (width, height)
        enable_smoothing: Whether to apply temporal smoothing
        smoothing_window: Number of frames for temporal smoothing
        batch_size: Batch size for processing multiple faces

    Example:
        >>> config = DetectorConfig(
        ...     confidence_threshold=0.7,
        ...     enable_smoothing=True,
        ...     smoothing_window=5
        ... )
        >>> detector = EmotionDetector(config=config)
    """

    # Model settings
    model_path: Optional[Union[str, Path]] = None
    cascade_path: Optional[Union[str, Path]] = None
    use_gpu: bool = False

    # Detection thresholds
    confidence_threshold: float = 0.5

    # Face detection parameters
    face_min_size: tuple[int, int] = (30, 30)
    face_scale_factor: float = 1.1
    face_min_neighbors: int = 5

    # Model input settings
    input_size: tuple[int, int] = (48, 48)

    # Smoothing settings
    enable_smoothing: bool = True
    smoothing_window: int = 5

    # Processing settings
    batch_size: int = 1

    def __post_init__(self) -> None:
        """Initialize and validate configuration after dataclass creation."""
        # Apply environment variable overrides
        self._apply_env_overrides()

        # Convert paths to Path objects
        if self.model_path is not None:
            self.model_path = Path(self.model_path)
        if self.cascade_path is not None:
            self.cascade_path = Path(self.cascade_path)

        # Validate configuration
        self._validate()

    def _apply_env_overrides(self) -> None:
        """Apply configuration overrides from environment variables."""
        if ENV_MODEL_PATH in os.environ and self.model_path is None:
            self.model_path = os.environ[ENV_MODEL_PATH]

        if ENV_CASCADE_PATH in os.environ and self.cascade_path is None:
            self.cascade_path = os.environ[ENV_CASCADE_PATH]

        if ENV_USE_GPU in os.environ:
            self.use_gpu = os.environ[ENV_USE_GPU].lower() in ("1", "true", "yes")

    def _validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.confidence_threshold}"
            )

        if self.face_scale_factor <= 1.0:
            raise ValueError(
                f"face_scale_factor must be greater than 1.0, "
                f"got {self.face_scale_factor}"
            )

        if self.face_min_neighbors < 1:
            raise ValueError(
                f"face_min_neighbors must be at least 1, "
                f"got {self.face_min_neighbors}"
            )

        if self.smoothing_window < 1:
            raise ValueError(
                f"smoothing_window must be at least 1, "
                f"got {self.smoothing_window}"
            )

        if self.batch_size < 1:
            raise ValueError(
                f"batch_size must be at least 1, got {self.batch_size}"
            )

        if any(dim < 10 for dim in self.input_size):
            raise ValueError(
                f"input_size dimensions must be at least 10, "
                f"got {self.input_size}"
            )

        if any(dim < 10 for dim in self.face_min_size):
            raise ValueError(
                f"face_min_size dimensions must be at least 10, "
                f"got {self.face_min_size}"
            )

    def get_model_path(self) -> Path:
        """Get the resolved model path.

        Returns:
            Path to the emotion recognition model

        Raises:
            FileNotFoundError: If no valid model path is configured
        """
        if self.model_path is not None:
            return self.model_path
        if DEFAULT_MODEL_PATH.exists():
            return DEFAULT_MODEL_PATH
        raise FileNotFoundError(
            "No model path configured. Set model_path or "
            f"the {ENV_MODEL_PATH} environment variable."
        )

    def get_cascade_path(self) -> Path:
        """Get the resolved cascade path.

        Returns:
            Path to the Haar cascade file

        Raises:
            FileNotFoundError: If no valid cascade path is configured
        """
        if self.cascade_path is not None:
            return self.cascade_path
        if DEFAULT_CASCADE_PATH.exists():
            return DEFAULT_CASCADE_PATH
        raise FileNotFoundError(
            "No cascade path configured. Set cascade_path or "
            f"the {ENV_CASCADE_PATH} environment variable."
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> DetectorConfig:
        """Create a configuration from a dictionary.

        Args:
            config_dict: Dictionary of configuration parameters

        Returns:
            DetectorConfig instance

        Example:
            >>> config = DetectorConfig.from_dict({
            ...     "confidence_threshold": 0.8,
            ...     "use_gpu": True
            ... })
        """
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "model_path": str(self.model_path) if self.model_path else None,
            "cascade_path": str(self.cascade_path) if self.cascade_path else None,
            "use_gpu": self.use_gpu,
            "confidence_threshold": self.confidence_threshold,
            "face_min_size": self.face_min_size,
            "face_scale_factor": self.face_scale_factor,
            "face_min_neighbors": self.face_min_neighbors,
            "input_size": self.input_size,
            "enable_smoothing": self.enable_smoothing,
            "smoothing_window": self.smoothing_window,
            "batch_size": self.batch_size,
        }

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return (
            f"DetectorConfig(threshold={self.confidence_threshold}, "
            f"gpu={self.use_gpu}, smoothing={self.enable_smoothing})"
        )
