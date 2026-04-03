"""
Annotator Module - Supervision-based Video Annotation
=====================================================

This module handles all visual annotation using the supervision library,
including bounding boxes, labels, trajectories, and confidence overlays.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import cv2
from pathlib import Path
from loguru import logger

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    logger.warning("supervision not available, using OpenCV fallback")


@dataclass
class AnnotationConfig:
    """Annotation configuration."""
    bbox_thickness: int = 2
    label_font_scale: float = 0.6
    label_font_thickness: int = 2
    trajectory_length: int = 30
    show_confidence: bool = True
    show_track_id: bool = True
    color_by_team: bool = False


class ColorPalette:
    """Color palette for track visualization."""
    
    # Default colors (BGR format)
    COLORS = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 255),    # Orange
        (255, 128, 0),    # Light Blue
        (0, 128, 255),    # Orange-Red
        (128, 255, 0),    # Light Green
        (255, 0, 128),    # Purple
        (0, 255, 128),    # Spring Green
    ]
    
    # Team colors
    TEAM_COLORS = {
        0: (255, 0, 0),    # Team 1: Blue
        1: (0, 0, 255),    # Team 2: Red
        2: (0, 255, 0),    # Team 3/Referee: Green
    }
    
    @classmethod
    def get_color(cls, track_id: int) -> Tuple[int, int, int]:
        """Get color for track ID."""
        return cls.COLORS[track_id % len(cls.COLORS)]
    
    @classmethod
    def get_team_color(cls, team_id: int) -> Tuple[int, int, int]:
        """Get color for team."""
        return cls.TEAM_COLORS.get(team_id, cls.COLORS[0])


class TrajectoryDrawer:
    """Draws track trajectories on frames."""
    
    def __init__(self, max_length: int = 30, line_thickness: int = 2):
        self.max_length = max_length
        self.line_thickness = line_thickness
        self.trajectories: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    
    def update(self, track_id: int, center: Tuple[float, float]):
        """Update trajectory for a track."""
        pos = (int(center[0]), int(center[1]))
        self.trajectories[track_id].append(pos)
        
        # Keep only recent positions
        if len(self.trajectories[track_id]) > self.max_length:
            self.trajectories[track_id] = self.trajectories[track_id][-self.max_length:]
    
    def draw(
        self,
        frame: np.ndarray,
        track_id: int,
        color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """Draw trajectory for a track."""
        if track_id not in self.trajectories:
            return frame
        
        positions = self.trajectories[track_id]
        if len(positions) < 2:
            return frame
        
        if color is None:
            color = ColorPalette.get_color(track_id)
        
        # Draw fading trajectory
        for i in range(1, len(positions)):
            alpha = i / len(positions)
            thickness = max(1, int(self.line_thickness * alpha))
            
            pt1 = positions[i - 1]
            pt2 = positions[i]
            
            # Fade color
            faded_color = tuple(int(c * alpha) for c in color)
            
            cv2.line(frame, pt1, pt2, faded_color, thickness)
        
        return frame
    
    def draw_all(
        self,
        frame: np.ndarray,
        active_tracks: List[int],
        team_assignments: Optional[Dict[int, int]] = None
    ) -> np.ndarray:
        """Draw all active trajectories."""
        for track_id in active_tracks:
            if team_assignments and track_id in team_assignments:
                color = ColorPalette.get_team_color(team_assignments[track_id])
            else:
                color = ColorPalette.get_color(track_id)
            
            frame = self.draw(frame, track_id, color)
        
        return frame
    
    def clear(self, track_id: Optional[int] = None):
        """Clear trajectory data."""
        if track_id is not None:
            self.trajectories.pop(track_id, None)
        else:
            self.trajectories.clear()


class VideoAnnotator:
    """
    Main video annotator using supervision library.
    
    Features:
    - Bounding box annotation with track IDs
    - Confidence score overlay
    - Trajectory visualization
    - Team-based coloring
    """
    
    def __init__(self, config: Optional[AnnotationConfig] = None):
        """
        Initialize annotator.
        
        Args:
            config: Annotation configuration
        """
        self.config = config or AnnotationConfig()
        
        # Initialize supervision annotators if available
        if SUPERVISION_AVAILABLE:
           self.box_annotator = sv.BoxAnnotator(
        thickness=self.config.bbox_thickness,
        color_lookup=sv.ColorLookup.INDEX
    )
        self.label_annotator = sv.LabelAnnotator(
        text_scale=self.config.label_font_scale,
        text_thickness=self.config.label_font_thickness,
        color_lookup=sv.ColorLookup.INDEX
    )
        self.trace_annotator = sv.TraceAnnotator(
        thickness=self.config.bbox_thickness,
        trace_length=self.config.trajectory_length,
        color_lookup=sv.ColorLookup.INDEX
    )
        
        # Trajectory drawer (fallback/additional)
        self.trajectory_drawer = TrajectoryDrawer(
            max_length=self.config.trajectory_length,
            line_thickness=self.config.bbox_thickness
        )
        
        # Team assignments
        self.team_assignments: Dict[int, int] = {}
        
        logger.success("VideoAnnotator initialized")
    
    def set_team_assignments(self, assignments: Dict[int, int]):
        """Set team assignments for color coding."""
        self.team_assignments = assignments
    
    def _get_color_for_track(self, track_id: int) -> Tuple[int, int, int]:
        """Get color based on track ID or team assignment."""
        if self.config.color_by_team and track_id in self.team_assignments:
            return ColorPalette.get_team_color(self.team_assignments[track_id])
        return ColorPalette.get_color(track_id)
    
    def annotate_frame(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        track_ids: List[int],
        confidences: Optional[np.ndarray] = None,
        draw_trajectories: bool = True,
        unified_ids: Optional[Dict[int, int]] = None
    ) -> np.ndarray:
        """
        Annotate a single frame with detections and tracks.
        
        Args:
            frame: Input frame (BGR)
            boxes: (N, 4) array of [x1, y1, x2, y2]
            track_ids: List of track IDs
            confidences: Optional confidence scores
            draw_trajectories: Whether to draw trajectory traces
            unified_ids: Optional ID mapping from ReID
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        if len(boxes) == 0:
            return annotated
        
        # Update trajectories
        for track_id, box in zip(track_ids, boxes):
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            self.trajectory_drawer.update(track_id, center)
        
        # Draw trajectories first (behind boxes)
        if draw_trajectories:
            annotated = self.trajectory_drawer.draw_all(
                annotated,
                track_ids,
                self.team_assignments if self.config.color_by_team else None
            )
        
        if SUPERVISION_AVAILABLE:
            # Use supervision for annotation
            detections = sv.Detections(
                xyxy=boxes.astype(np.float32),
                confidence=confidences if confidences is not None else np.ones(len(boxes)),
                tracker_id=np.array(track_ids)
            )
            
            # Build labels
            labels = []
            for i, track_id in enumerate(track_ids):
                display_id = unified_ids.get(track_id, track_id) if unified_ids else track_id
                label = f"ID:{display_id}"
                
                if self.config.show_confidence and confidences is not None:
                    label += f" {confidences[i]:.2f}"
                
                labels.append(label)
            
            # Annotate with boxes and labels
            annotated = self.box_annotator.annotate(annotated, detections)
            annotated = self.label_annotator.annotate(annotated, detections, labels)
        else:
            # Fallback to OpenCV annotation
            annotated = self._annotate_opencv(
                annotated, boxes, track_ids, confidences, unified_ids
            )
        
        return annotated
    
    def _annotate_opencv(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        track_ids: List[int],
        confidences: Optional[np.ndarray],
        unified_ids: Optional[Dict[int, int]]
    ) -> np.ndarray:
        """OpenCV fallback annotation."""
        for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Get color
            color = self._get_color_for_track(track_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.bbox_thickness)
            
            # Build label
            display_id = unified_ids.get(track_id, track_id) if unified_ids else track_id
            label = f"ID:{display_id}"
            
            if self.config.show_confidence and confidences is not None:
                label += f" {confidences[i]:.2f}"
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.label_font_scale,
                self.config.label_font_thickness
            )
            
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.label_font_scale,
                (255, 255, 255),
                self.config.label_font_thickness
            )
        
        return frame
    
    def add_info_overlay(
        self,
        frame: np.ndarray,
        frame_id: int,
        num_tracks: int,
        fps: Optional[float] = None,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Add information overlay to frame.
        
        Args:
            frame: Input frame
            frame_id: Frame number
            num_tracks: Number of active tracks
            fps: Processing FPS
            extra_info: Additional info to display
            
        Returns:
            Frame with overlay
        """
        h, w = frame.shape[:2]
        
        # Info text
        info_lines = [
            f"Frame: {frame_id}",
            f"Tracks: {num_tracks}"
        ]
        
        if fps is not None:
            info_lines.append(f"FPS: {fps:.1f}")
        
        if extra_info:
            for key, value in extra_info.items():
                info_lines.append(f"{key}: {value}")
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (200, 30 + 25 * len(info_lines)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw text
        for i, line in enumerate(info_lines):
            cv2.putText(
                frame,
                line,
                (20, 35 + 25 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
        
        return frame


class VideoWriter:
    """Video writer with Streamlit-compatible output."""
    
    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = None
    ):
        """
        Initialize video writer.
        
        Args:
            output_path: Output video path
            fps: Frame rate
            width: Frame width
            height: Frame height
            codec: Video codec (auto-detect based on extension if None)
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        
        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # For MP4 files, use codecs that work with Streamlit (H.264)
        # Try multiple codecs in order of compatibility
        ext = self.output_path.suffix.lower()
        if codec:
            codecs_to_try = [codec]
        elif ext == ".mp4":
            # These codecs produce Streamlit-compatible H.264 MP4
            codecs_to_try = ["avc1", "H264", "X264", "mp4v", "XVID"]
        elif ext == ".avi":
            codecs_to_try = ["XVID", "MJPG", "DIVX"]
        else:
            codecs_to_try = ["mp4v", "XVID", "MJPG"]
        
        self.writer = None
        used_codec = None
        for codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                self.writer = cv2.VideoWriter(
                    str(self.output_path),
                    fourcc,
                    fps,
                    (width, height)
                )
                if self.writer.isOpened():
                    used_codec = codec_name
                    break
                self.writer.release()
                self.writer = None
            except Exception:
                continue
        
        if self.writer is None or not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {output_path}")
        
        self.frame_count = 0
        self.used_codec = used_codec
        logger.info(f"VideoWriter initialized: {output_path} ({width}x{height} @ {fps}fps, codec={used_codec})")
    
    def write(self, frame: np.ndarray):
        """Write a frame to video."""
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def release(self):
        """Release video writer."""
        if self.writer is not None:
            self.writer.release()
            logger.success(f"Video saved: {self.output_path} ({self.frame_count} frames)")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def reencode_video(
    input_path: str,
    output_path: str,
    codec: str = "libx264",
    crf: int = 23
) -> bool:
    """
    Re-encode video using ffmpeg for better compatibility.
    
    Args:
        input_path: Input video path
        output_path: Output video path
        codec: Video codec
        crf: Constant rate factor (quality)
        
    Returns:
        Success status
    """
    try:
        import ffmpeg
        
        (
            ffmpeg
            .input(input_path)
            .output(output_path, vcodec=codec, crf=crf)
            .overwrite_output()
            .run(quiet=True)
        )
        
        logger.success(f"Video re-encoded: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Re-encoding failed: {e}")
        return False
