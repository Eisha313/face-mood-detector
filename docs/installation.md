# Installation Guide

## Requirements

- Python 3.8 or higher
- OpenCV-compatible webcam (for real-time detection)
- CUDA-compatible GPU (optional, for faster inference)

## Installation Methods

### Using pip (Recommended)

```bash
pip install face-mood-detector
```

### From Source

```bash
git clone https://github.com/example/face-mood-detector.git
cd face-mood-detector
pip install -e .
```

### Development Installation

For contributing or development purposes:

```bash
git clone https://github.com/example/face-mood-detector.git
cd face-mood-detector
pip install -e ".[dev]"
```

Or using requirements files:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|--------|
| numpy | >=1.21.0 | Array operations |
| opencv-python | >=4.5.0 | Image processing and webcam access |
| tensorflow | >=2.8.0 | Deep learning inference |
| Pillow | >=8.0.0 | Image loading utilities |

### Optional Dependencies

#### GPU Support

For NVIDIA GPU acceleration:

```bash
pip install tensorflow-gpu>=2.8.0
```

Make sure you have:
- NVIDIA drivers installed
- CUDA Toolkit 11.2+
- cuDNN 8.1+

#### Visualization

For advanced visualization and reporting:

```bash
pip install matplotlib>=3.4.0 pandas>=1.3.0
```

## Verifying Installation

After installation, verify everything works:

```python
import face_mood_detector

# Check version
print(f"Version: {face_mood_detector.__version__}")

# Quick test
from face_mood_detector import EmotionDetector
detector = EmotionDetector()
print("Installation successful!")
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'cv2'

```bash
pip install opencv-python-headless  # For servers without display
# or
pip install opencv-python  # For desktop with GUI support
```

#### TensorFlow GPU not detected

1. Verify CUDA installation:
```bash
nvcc --version
```

2. Check TensorFlow GPU access:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

#### Webcam not accessible

- Ensure no other application is using the webcam
- Check webcam permissions (especially on Linux/macOS)
- Try different camera indices: `VideoAnalyzer(camera_index=1)`

### Platform-Specific Notes

#### Linux

You may need to install system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

#### macOS

Grant camera permissions in System Preferences > Security & Privacy > Camera.

#### Windows

Ensure Visual C++ Redistributable is installed for TensorFlow.

## Next Steps

- Read the [Quickstart Guide](quickstart.md) to start detecting emotions
- Check [API Reference](api.md) for detailed documentation
- See [Examples](examples.md) for common use cases
