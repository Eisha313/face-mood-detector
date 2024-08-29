# Face Mood Detector API Reference

## Core Classes

### EmotionDetector

The main class for performing emotion detection on images.

```python
from face_mood_detector import EmotionDetector

detector = EmotionDetector(
    model_path=None,  # Path to custom model weights
    confidence_threshold=0.5,  # Minimum confidence for predictions
    device='auto'  # 'cpu', 'cuda', or 'auto'
)
```

#### Methods

##### `detect(image, return_scores=False)`

Detect emotions in a single image.

**Parameters:**
- `image`: numpy array (BGR or RGB format) or path to image file
- `return_scores`: If True, returns confidence scores for all emotions

**Returns:**
- `EmotionResult` object containing detected faces and emotions

```python
result = detector.detect('photo.jpg')
for face in result.faces:
    print(f"Emotion: {face.emotion}, Confidence: {face.confidence:.2f}")
```

##### `detect_batch(images)`

Detect emotions in multiple images efficiently.

**Parameters:**
- `images`: List of numpy arrays or image paths

**Returns:**
- List of `EmotionResult` objects

---

### VideoAnalyzer

Process video streams for real-time emotion detection.

```python
from face_mood_detector import VideoAnalyzer

analyzer = VideoAnalyzer(
    detector=None,  # Custom EmotionDetector instance
    smoothing_window=5,  # Frames for temporal smoothing
    fps_limit=30  # Maximum processing FPS
)
```

#### Methods

##### `start_webcam(camera_id=0, callback=None)`

Start real-time emotion detection from webcam.

**Parameters:**
- `camera_id`: Camera device index
- `callback`: Function called with each frame's results

```python
def on_frame(results):
    for face in results.faces:
        print(f"{face.emotion}: {face.confidence:.2f}")

analyzer.start_webcam(callback=on_frame)
```

##### `process_video(video_path, output_path=None)`

Process a video file and optionally save annotated output.

**Parameters:**
- `video_path`: Path to input video
- `output_path`: Optional path for annotated output video

**Returns:**
- `VideoAnalysisResult` with frame-by-frame emotion data

##### `stop()`

Stop ongoing video processing.

---

### TemporalSmoother

Smooth emotion predictions over time for stable results.

```python
from face_mood_detector.smoother import TemporalSmoother

smoother = TemporalSmoother(
    window_size=5,  # Number of frames to average
    method='exponential'  # 'simple', 'exponential', or 'median'
)
```

#### Methods

##### `smooth(scores)`

Add new scores and get smoothed result.

**Parameters:**
- `scores`: Dict of emotion -> confidence score

**Returns:**
- Smoothed emotion scores dict

##### `reset()`

Clear the smoothing buffer.

---

## Data Classes

### Emotion

Enum of supported emotions:

```python
from face_mood_detector.emotions import Emotion

Emotion.HAPPY
Emotion.SAD
Emotion.ANGRY
Emotion.FEARFUL
Emotion.DISGUSTED
Emotion.SURPRISED
Emotion.NEUTRAL
```

### EmotionResult

Container for detection results:

```python
result.faces  # List of FaceEmotion objects
result.image  # Annotated image (if requested)
result.timestamp  # Detection timestamp
```

### FaceEmotion

Emotion data for a single face:

```python
face.emotion  # Predicted Emotion enum
face.confidence  # Confidence score (0-1)
face.bbox  # Bounding box (x, y, width, height)
face.all_scores  # Dict of all emotion scores
```

---

## Configuration

### DetectorConfig

Global configuration options:

```python
from face_mood_detector.config import DetectorConfig

config = DetectorConfig(
    face_detection_scale=1.0,
    min_face_size=48,
    emotion_model_input_size=(48, 48),
    use_gpu=True
)
```

---

## Preprocessing

Utilities for image preprocessing:

```python
from face_mood_detector.preprocessing import (
    normalize_image,
    resize_face,
    convert_to_grayscale,
    augment_image
)

# Prepare image for model
processed = normalize_image(face_crop)
resized = resize_face(processed, target_size=(48, 48))
```
