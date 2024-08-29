# Face Mood Detector

A lightweight Python library for real-time facial emotion recognition using CNN-based deep learning.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🎭 **Real-time emotion detection** from webcam feed with 7 basic emotion classifications
- 🧠 **Pre-trained CNN model** with fine-tuning capabilities on custom datasets
- 🔌 **Simple API** for single image and video stream emotion analysis
- 📊 **Confidence scores** and temporal smoothing for stable predictions
- 📈 **Export utilities** for emotion analytics and visualization reports

## Supported Emotions

- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fearful
- 🤢 Disgusted
- 😲 Surprised
- 😐 Neutral

## Installation

```bash
pip install face-mood-detector
```

Or install from source:

```bash
git clone https://github.com/username/face-mood-detector.git
cd face-mood-detector
pip install -e .
```

## Quick Start

### Single Image Analysis

```python
from face_mood_detector import EmotionDetector

detector = EmotionDetector()
result = detector.detect('path/to/photo.jpg')

for face in result.faces:
    print(f"Emotion: {face.emotion.value}, Confidence: {face.confidence:.2f}")
```

### Real-time Webcam Detection

```python
from face_mood_detector import VideoAnalyzer

analyzer = VideoAnalyzer(smoothing_window=5)
analyzer.start_webcam()
```

### Process Video File

```python
from face_mood_detector import VideoAnalyzer

analyzer = VideoAnalyzer()
results = analyzer.process_video('input.mp4', output_path='output.mp4')
```

## Documentation

- [API Reference](docs/api.md)
- [Examples](docs/examples.md)

## Configuration

Customize detection parameters:

```python
from face_mood_detector import EmotionDetector
from face_mood_detector.config import DetectorConfig

config = DetectorConfig(
    min_face_size=48,
    use_gpu=True
)

detector = EmotionDetector(
    confidence_threshold=0.6,
    device='cuda'
)
```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 face_mood_detector/
mypy face_mood_detector/
```

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- NumPy

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Acknowledgments

- FER2013 dataset for initial model training
- OpenCV for face detection capabilities
- PyTorch team for the deep learning framework
