"""Tests for video analyzer module."""

import numpy as np
import pytest

from face_mood_detector.video_analyzer import VideoAnalyzer, FrameResult
from face_mood_detector.emotions import Emotion, EmotionResult


class TestVideoAnalyzer:
    """Tests for VideoAnalyzer class."""
    
    def test_init_default(self):
        """Test default initialization."""
        analyzer = VideoAnalyzer()
        
        assert analyzer.detector is not None
        assert analyzer.draw_overlay is True
        assert analyzer.show_confidence is True
        assert analyzer.box_color == (0, 255, 0)
        assert analyzer.text_color == (255, 255, 255)
    
    def test_init_custom_options(self):
        """Test initialization with custom options."""
        analyzer = VideoAnalyzer(
            draw_overlay=False,
            show_confidence=False,
            box_color=(255, 0, 0),
            text_color=(0, 0, 255),
            font_scale=1.0,
            box_thickness=3
        )
        
        assert analyzer.draw_overlay is False
        assert analyzer.show_confidence is False
        assert analyzer.box_color == (255, 0, 0)
        assert analyzer.text_color == (0, 0, 255)
        assert analyzer.font_scale == 1.0
        assert analyzer.box_thickness == 3
    
    def test_process_frame(self):
        """Test processing a single frame."""
        analyzer = VideoAnalyzer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        output, emotions = analyzer.process_frame(frame)
        
        assert output.shape == frame.shape
        assert isinstance(emotions, list)
    
    def test_process_frame_no_overlay(self):
        """Test processing without overlay."""
        analyzer = VideoAnalyzer(draw_overlay=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        output, emotions = analyzer.process_frame(frame)
        
        assert output.shape == frame.shape
        # Without overlay, output should be a copy of input
        assert np.array_equal(output, frame)
    
    def test_draw_detection_overlay(self):
        """Test drawing overlay on frame."""
        analyzer = VideoAnalyzer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        emotions = [
            EmotionResult(
                emotion=Emotion.HAPPY,
                confidence=0.85,
                all_confidences={e: 0.1 for e in Emotion},
                bounding_box={'x': 100, 'y': 100, 'width': 100, 'height': 100}
            )
        ]
        emotions[0].all_confidences[Emotion.HAPPY] = 0.85
        
        output = analyzer._draw_detection_overlay(frame, emotions)
        
        assert output.shape == frame.shape
        # Output should be different from input (has overlay)
        assert not np.array_equal(output, frame)
    
    def test_draw_fps(self):
        """Test drawing FPS counter."""
        analyzer = VideoAnalyzer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        output = analyzer._draw_fps(frame, 30.0)
        
        assert output.shape == frame.shape
        # Output should be modified with FPS text
        assert not np.array_equal(output, frame)
    
    def test_stop(self):
        """Test stopping the analyzer."""
        analyzer = VideoAnalyzer()
        analyzer._running = True
        
        analyzer.stop()
        
        assert analyzer.is_running is False
    
    def test_is_running_property(self):
        """Test is_running property."""
        analyzer = VideoAnalyzer()
        
        assert analyzer.is_running is False
        
        analyzer._running = True
        assert analyzer.is_running is True
    
    def test_frame_count_property(self):
        """Test frame_count property."""
        analyzer = VideoAnalyzer()
        
        assert analyzer.frame_count == 0
        
        analyzer._frame_count = 100
        assert analyzer.frame_count == 100


class TestFrameResult:
    """Tests for FrameResult dataclass."""
    
    def test_frame_result_creation(self):
        """Test creating a FrameResult."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emotions = []
        
        result = FrameResult(
            frame=frame,
            emotions=emotions,
            timestamp=1.5,
            frame_number=45,
            fps=30.0
        )
        
        assert np.array_equal(result.frame, frame)
        assert result.emotions == []
        assert result.timestamp == 1.5
        assert result.frame_number == 45
        assert result.fps == 30.0
