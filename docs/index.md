# Face Mood Detector

**A lightweight Python library for real-time facial emotion recognition using CNN-based deep learning.**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Features

- 🎥 **Real-time Detection**: Process webcam feeds with minimal latency
- 🎯 **7 Emotion Classes**: Detect angry, disgust, fear, happy, sad, surprise, and neutral
- 🧠 **Pre-trained CNN**: Ready to use out of the box, with fine-tuning support
- 📊 **Confidence Scores**: Get probability scores for all emotions
- 🔄 **Temporal Smoothing**: Stable predictions for video streams
- 📈 **Analytics Export**: Generate reports and visualizations

## Quick Example

```python
from face_mood_detector import EmotionDetector

# Initialize and detect
detector = EmotionDetector()
result = detector.detect("happy_face.jpg")

print(f"Emotion: {result[0]['emotion']}")
print(f"Confidence: {result[0]['confidence']:.2%}")
```

## Real-time Webcam

```python
from face_mood_detector import VideoAnalyzer

analyzer = VideoAnalyzer()
analyzer.run_live(display=True)  # Press 'q' to quit
```

## Documentation

- [Installation Guide](installation.md) - Setup and requirements
- [Quickstart](quickstart.md) - Get up and running quickly
- [API Reference](api.md) - Detailed API documentation
- [Examples](examples.md) - Common use cases and patterns

## Architecture

```
face_mood_detector/
├── detector.py        # Main EmotionDetector class
├── video_analyzer.py  # Video/webcam processing
├── face_detector.py   # Face detection utilities
├── smoother.py        # Temporal smoothing
├── preprocessing.py   # Image preprocessing
├── emotions.py        # Emotion definitions
└── config.py          # Configuration options
```

## Performance

| Hardware | FPS (640x480) | FPS (1080p) |
|----------|---------------|-------------|
| CPU (i7-10700) | ~15 | ~8 |
| GPU (RTX 3060) | ~45 | ~30 |
| GPU (RTX 3090) | ~60 | ~45 |

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

MIT License - see [LICENSE](../LICENSE) for details.
