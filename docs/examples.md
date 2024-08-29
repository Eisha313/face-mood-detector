# Face Mood Detector Examples

## Quick Start

### Basic Image Analysis

```python
from face_mood_detector import EmotionDetector

# Initialize detector
detector = EmotionDetector()

# Analyze a single image
result = detector.detect('path/to/photo.jpg')

# Print results
for face in result.faces:
    print(f"Detected {face.emotion.value} with {face.confidence:.1%} confidence")
```

### Real-time Webcam Detection

```python
from face_mood_detector import EmotionDetector, VideoAnalyzer
import cv2

detector = EmotionDetector()
analyzer = VideoAnalyzer(detector=detector, smoothing_window=10)

def display_callback(results, frame):
    # Draw results on frame
    for face in results.faces:
        x, y, w, h = face.bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{face.emotion.value}: {face.confidence:.0%}",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Emotion Detection', frame)
    return cv2.waitKey(1) != ord('q')

analyzer.start_webcam(callback=display_callback)
cv2.destroyAllWindows()
```

### Process Video File

```python
from face_mood_detector import VideoAnalyzer

analyzer = VideoAnalyzer()

# Process video and save annotated output
results = analyzer.process_video(
    'input_video.mp4',
    output_path='output_annotated.mp4'
)

# Analyze emotion timeline
for frame_data in results.timeline:
    print(f"Frame {frame_data.frame_number}: {frame_data.dominant_emotion}")
```

---

## Advanced Usage

### Custom Confidence Threshold

```python
from face_mood_detector import EmotionDetector

# Only return high-confidence predictions
detector = EmotionDetector(confidence_threshold=0.7)

result = detector.detect(image)
# Only faces with emotion confidence >= 70% are included
```

### Getting All Emotion Scores

```python
result = detector.detect(image, return_scores=True)

for face in result.faces:
    print(f"Primary emotion: {face.emotion.value}")
    print("All scores:")
    for emotion, score in face.all_scores.items():
        print(f"  {emotion}: {score:.2%}")
```

### Temporal Smoothing for Stable Video

```python
from face_mood_detector import EmotionDetector
from face_mood_detector.smoother import TemporalSmoother

detector = EmotionDetector()
smoother = TemporalSmoother(window_size=10, method='exponential')

# In your video processing loop:
result = detector.detect(frame, return_scores=True)

if result.faces:
    raw_scores = result.faces[0].all_scores
    smoothed_scores = smoother.smooth(raw_scores)
    
    # smoothed_scores provides more stable predictions
    stable_emotion = max(smoothed_scores, key=smoothed_scores.get)
```

### Batch Processing Multiple Images

```python
import os
from face_mood_detector import EmotionDetector

detector = EmotionDetector()

# Get all images in a folder
image_folder = 'path/to/images'
images = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
          if f.endswith(('.jpg', '.png', '.jpeg'))]

# Process in batch (more efficient than one at a time)
results = detector.detect_batch(images)

for image_path, result in zip(images, results):
    print(f"{image_path}: {len(result.faces)} faces detected")
```

### Using with NumPy Arrays

```python
import cv2
import numpy as np
from face_mood_detector import EmotionDetector

detector = EmotionDetector()

# From OpenCV
img = cv2.imread('photo.jpg')
result = detector.detect(img)

# From PIL
from PIL import Image
pil_img = Image.open('photo.jpg')
np_img = np.array(pil_img)
result = detector.detect(np_img)
```

---

## Integration Examples

### Flask Web API

```python
from flask import Flask, request, jsonify
from face_mood_detector import EmotionDetector
import numpy as np
import cv2

app = Flask(__name__)
detector = EmotionDetector()

@app.route('/analyze', methods=['POST'])
def analyze_emotion():
    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    result = detector.detect(img, return_scores=True)
    
    faces = []
    for face in result.faces:
        faces.append({
            'emotion': face.emotion.value,
            'confidence': face.confidence,
            'bbox': face.bbox,
            'scores': {e.value: s for e, s in face.all_scores.items()}
        })
    
    return jsonify({'faces': faces})

if __name__ == '__main__':
    app.run(debug=True)
```

### Emotion Statistics Collection

```python
from collections import defaultdict
from face_mood_detector import VideoAnalyzer

analyzer = VideoAnalyzer()
emotion_counts = defaultdict(int)

def collect_stats(results, frame):
    for face in results.faces:
        emotion_counts[face.emotion.value] += 1
    return True

# Process video
analyzer.process_video('meeting_recording.mp4')

# Print statistics
total = sum(emotion_counts.values())
print("Emotion Distribution:")
for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
    print(f"  {emotion}: {count/total:.1%}")
```
