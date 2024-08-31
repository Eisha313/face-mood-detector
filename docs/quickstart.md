# Quickstart Guide

Get started with face-mood-detector in minutes!

## Basic Usage

### Detecting Emotions in an Image

```python
from face_mood_detector import EmotionDetector

# Initialize the detector
detector = EmotionDetector()

# Analyze an image
results = detector.detect("path/to/image.jpg")

# Print results
for face in results:
    print(f"Emotion: {face['emotion']}")
    print(f"Confidence: {face['confidence']:.2f}")
    print(f"All scores: {face['scores']}")
```

### Real-time Webcam Detection

```python
from face_mood_detector import VideoAnalyzer

# Create analyzer with webcam
analyzer = VideoAnalyzer()

# Start real-time detection
# Press 'q' to quit
analyzer.run_live(display=True)
```

### Processing a Video File

```python
from face_mood_detector import VideoAnalyzer

analyzer = VideoAnalyzer()

# Process video and get frame-by-frame results
results = analyzer.process_video(
    "path/to/video.mp4",
    output_path="annotated_output.mp4"  # Optional: save annotated video
)

# Analyze the results
for frame_result in results:
    print(f"Frame {frame_result['frame']}: {frame_result['emotions']}")
```

## Understanding Results

### Emotion Labels

The detector recognizes 7 basic emotions:

| Emotion | Description |
|---------|-------------|
| `angry` | Anger, frustration |
| `disgust` | Disgust, distaste |
| `fear` | Fear, anxiety |
| `happy` | Happiness, joy |
| `sad` | Sadness, sorrow |
| `surprise` | Surprise, shock |
| `neutral` | Neutral, calm |

### Result Structure

```python
# Single detection result
{
    'emotion': 'happy',           # Dominant emotion
    'confidence': 0.92,           # Confidence score (0-1)
    'scores': {                   # All emotion scores
        'angry': 0.01,
        'disgust': 0.00,
        'fear': 0.02,
        'happy': 0.92,
        'sad': 0.01,
        'surprise': 0.03,
        'neutral': 0.01
    },
    'bbox': (x, y, width, height)  # Face bounding box
}
```

## Configuration Options

### Detector Settings

```python
from face_mood_detector import EmotionDetector
from face_mood_detector.config import DetectorConfig

# Custom configuration
config = DetectorConfig(
    min_confidence=0.5,      # Minimum confidence threshold
    face_min_size=30,        # Minimum face size in pixels
    use_gpu=True,            # Enable GPU acceleration
    batch_size=8             # Batch size for multiple faces
)

detector = EmotionDetector(config=config)
```

### Temporal Smoothing

For stable predictions in video streams:

```python
from face_mood_detector import VideoAnalyzer
from face_mood_detector.smoother import EmotionSmoother

# Configure smoothing
smoother = EmotionSmoother(
    window_size=5,    # Number of frames to average
    method='ema'      # Exponential moving average
)

analyzer = VideoAnalyzer(smoother=smoother)
```

## Working with NumPy Arrays

```python
import cv2
import numpy as np
from face_mood_detector import EmotionDetector

detector = EmotionDetector()

# Load image as numpy array
image = cv2.imread("photo.jpg")
# or create from webcam frame, PIL image, etc.

# Detect emotions
results = detector.detect(image)
```

## Multiple Faces

The detector automatically handles multiple faces:

```python
results = detector.detect("group_photo.jpg")

print(f"Found {len(results)} faces")
for i, face in enumerate(results):
    print(f"Face {i+1}: {face['emotion']} ({face['confidence']:.0%})")
```

## Error Handling

```python
from face_mood_detector import EmotionDetector
from face_mood_detector.exceptions import NoFaceDetectedError

detector = EmotionDetector()

try:
    results = detector.detect("image.jpg")
    if not results:
        print("No faces detected in the image")
    else:
        print(f"Detected: {results[0]['emotion']}")
except FileNotFoundError:
    print("Image file not found")
except Exception as e:
    print(f"Error: {e}")
```

## Next Steps

- Explore the [API Reference](api.md) for complete documentation
- See [Examples](examples.md) for advanced use cases
- Learn about [fine-tuning](examples.md#fine-tuning) on custom datasets
