"""Video stream analyzer for real-time emotion detection."""

import time
from typing import Callable, Optional, Generator
from dataclasses import dataclass

import cv2
import numpy as np

from .detector import EmotionDetector
from .emotions import EmotionResult


@dataclass
class FrameResult:
    """Result for a single video frame."""
    
    frame: np.ndarray
    emotions: list[EmotionResult]
    timestamp: float
    frame_number: int
    fps: float


class VideoAnalyzer:
    """Analyzer for video streams and webcam feeds.
    
    Provides real-time emotion detection from video sources
    with optional visualization overlay.
    """
    
    def __init__(
        self,
        detector: Optional[EmotionDetector] = None,
        draw_overlay: bool = True,
        show_confidence: bool = True,
        box_color: tuple[int, int, int] = (0, 255, 0),
        text_color: tuple[int, int, int] = (255, 255, 255),
        font_scale: float = 0.7,
        box_thickness: int = 2
    ):
        """Initialize the video analyzer.
        
        Args:
            detector: EmotionDetector instance. Creates new one if None.
            draw_overlay: Whether to draw detection overlay on frames.
            show_confidence: Whether to show confidence scores in overlay.
            box_color: BGR color for bounding boxes.
            text_color: BGR color for text labels.
            font_scale: Scale factor for text size.
            box_thickness: Thickness of bounding box lines.
        """
        self.detector = detector or EmotionDetector()
        self.draw_overlay = draw_overlay
        self.show_confidence = show_confidence
        self.box_color = box_color
        self.text_color = text_color
        self.font_scale = font_scale
        self.box_thickness = box_thickness
        
        self._running = False
        self._frame_count = 0
        self._start_time = 0.0
    
    def _draw_detection_overlay(
        self,
        frame: np.ndarray,
        emotions: list[EmotionResult]
    ) -> np.ndarray:
        """Draw emotion detection overlay on frame.
        
        Args:
            frame: Input video frame.
            emotions: List of emotion results to draw.
            
        Returns:
            Frame with overlay drawn.
        """
        output = frame.copy()
        
        for result in emotions:
            bbox = result.bounding_box
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            
            # Draw bounding box
            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                self.box_color,
                self.box_thickness
            )
            
            # Prepare label text
            if self.show_confidence:
                label = f"{result.emotion.value}: {result.confidence:.1%}"
            else:
                label = result.emotion.value
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                1
            )
            
            # Draw text background
            cv2.rectangle(
                output,
                (x, y - text_height - 10),
                (x + text_width + 5, y),
                self.box_color,
                -1
            )
            
            # Draw text
            cv2.putText(
                output,
                label,
                (x + 2, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.text_color,
                1,
                cv2.LINE_AA
            )
        
        return output
    
    def _draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter on frame.
        
        Args:
            frame: Input video frame.
            fps: Current frames per second.
            
        Returns:
            Frame with FPS counter.
        """
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        return frame
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[EmotionResult]]:
        """Process a single video frame.
        
        Args:
            frame: Input video frame in BGR format.
            
        Returns:
            Tuple of (processed frame, emotion results).
        """
        # Detect emotions
        emotions = self.detector.detect(frame)
        
        # Draw overlay if enabled
        if self.draw_overlay:
            output = self._draw_detection_overlay(frame, emotions)
        else:
            output = frame.copy()
        
        return output, emotions
    
    def analyze_stream(
        self,
        source: int | str = 0,
        max_frames: Optional[int] = None,
        skip_frames: int = 0
    ) -> Generator[FrameResult, None, None]:
        """Analyze a video stream frame by frame.
        
        Args:
            source: Video source (0 for webcam, or path to video file).
            max_frames: Maximum number of frames to process.
            skip_frames: Number of frames to skip between detections.
            
        Yields:
            FrameResult for each processed frame.
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        self._running = True
        self._frame_count = 0
        self._start_time = time.time()
        skip_counter = 0
        last_emotions: list[EmotionResult] = []
        
        try:
            while self._running:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                self._frame_count += 1
                
                # Skip frames if configured
                if skip_counter < skip_frames:
                    skip_counter += 1
                    # Use last emotions for skipped frames
                    if self.draw_overlay:
                        frame = self._draw_detection_overlay(frame, last_emotions)
                else:
                    frame, last_emotions = self.process_frame(frame)
                    skip_counter = 0
                
                # Calculate FPS
                elapsed = time.time() - self._start_time
                fps = self._frame_count / elapsed if elapsed > 0 else 0.0
                
                yield FrameResult(
                    frame=frame,
                    emotions=last_emotions,
                    timestamp=elapsed,
                    frame_number=self._frame_count,
                    fps=fps
                )
                
                if max_frames and self._frame_count >= max_frames:
                    break
        finally:
            cap.release()
            self._running = False
    
    def run_webcam(
        self,
        camera_id: int = 0,
        window_name: str = "Face Mood Detector",
        show_fps: bool = True,
        on_frame: Optional[Callable[[FrameResult], None]] = None,
        skip_frames: int = 0
    ) -> None:
        """Run live webcam emotion detection with display window.
        
        Args:
            camera_id: Camera device ID.
            window_name: Name for the display window.
            show_fps: Whether to display FPS counter.
            on_frame: Optional callback for each frame result.
            skip_frames: Number of frames to skip between detections.
        """
        print(f"Starting webcam feed (camera {camera_id})...")
        print("Press 'q' to quit")
        
        try:
            for result in self.analyze_stream(camera_id, skip_frames=skip_frames):
                frame = result.frame
                
                if show_fps:
                    frame = self._draw_fps(frame, result.fps)
                
                cv2.imshow(window_name, frame)
                
                if on_frame:
                    on_frame(result)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break
        finally:
            cv2.destroyAllWindows()
    
    def stop(self) -> None:
        """Stop the video analysis."""
        self._running = False
    
    @property
    def is_running(self) -> bool:
        """Check if analyzer is currently running."""
        return self._running
    
    @property
    def frame_count(self) -> int:
        """Get the current frame count."""
        return self._frame_count
