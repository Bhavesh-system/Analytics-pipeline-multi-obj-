"""
Analytics Module - Comprehensive Visual Analytics
=================================================

This module implements all analytics enhancements:
- Movement heatmaps
- Bird's-eye view projection
- Team clustering (jersey color)
- Speed estimation
- Object count over time
- Evaluation metrics
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import cv2
from pathlib import Path
import json
from loguru import logger

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class TrackStats:
    """Statistics for a single track."""
    track_id: int
    positions: List[Tuple[float, float]] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)
    speeds: List[float] = field(default_factory=list)
    team_id: Optional[int] = None
    total_distance: float = 0.0
    avg_speed: float = 0.0
    max_speed: float = 0.0
    
    def add_position(self, x: float, y: float, frame_id: int):
        self.positions.append((x, y))
        self.frame_ids.append(frame_id)


class HeatmapGenerator:
    """Generate movement heatmaps from track positions."""
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (100, 100),
        gaussian_sigma: float = 5.0,
        colormap: str = "jet",
        alpha: float = 0.6
    ):
        self.resolution = resolution
        self.gaussian_sigma = gaussian_sigma
        self.colormap = colormap
        self.alpha = alpha
        
        # Accumulator
        self.heatmap = np.zeros(resolution, dtype=np.float32)
        self.frame_size: Optional[Tuple[int, int]] = None
    
    def set_frame_size(self, width: int, height: int):
        """Set the frame size for coordinate mapping."""
        self.frame_size = (width, height)
    
    def add_position(self, x: float, y: float):
        """Add a position to the heatmap."""
        if self.frame_size is None:
            return
        
        # Map to heatmap coordinates
        hx = int(x / self.frame_size[0] * self.resolution[0])
        hy = int(y / self.frame_size[1] * self.resolution[1])
        
        # Clamp to bounds
        hx = max(0, min(self.resolution[0] - 1, hx))
        hy = max(0, min(self.resolution[1] - 1, hy))
        
        self.heatmap[hy, hx] += 1
    
    def add_positions(self, positions: List[Tuple[float, float]]):
        """Add multiple positions."""
        for x, y in positions:
            self.add_position(x, y)
    
    def generate(self) -> np.ndarray:
        """Generate smoothed heatmap."""
        if SCIPY_AVAILABLE:
            smoothed = gaussian_filter(self.heatmap, sigma=self.gaussian_sigma)
        else:
            # Fallback: simple box filter
            kernel_size = int(self.gaussian_sigma * 2) | 1
            smoothed = cv2.blur(self.heatmap, (kernel_size, kernel_size))
        
        # Normalize
        if smoothed.max() > 0:
            smoothed = smoothed / smoothed.max()
        
        return smoothed
    
    def get_overlay(
        self,
        frame: np.ndarray,
        colormap: Optional[str] = None
    ) -> np.ndarray:
        """
        Get heatmap overlay on frame.
        
        Args:
            frame: Background frame
            colormap: Optional colormap override
            
        Returns:
            Frame with heatmap overlay
        """
        heatmap = self.generate()
        
        # Resize to frame size
        h, w = frame.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Apply colormap
        cmap = colormap or self.colormap
        if MATPLOTLIB_AVAILABLE:
            cm = plt.get_cmap(cmap)
            heatmap_colored = (cm(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
        else:
            heatmap_colored = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
        
        # Blend with frame
        mask = heatmap_resized > 0.05
        overlay = frame.copy()
        overlay[mask] = cv2.addWeighted(
            frame[mask], 1 - self.alpha,
            heatmap_colored[mask], self.alpha, 0
        )
        
        return overlay
    
    def save_heatmap(self, output_path: str, background: Optional[np.ndarray] = None):
        """Save heatmap as image."""
        heatmap = self.generate()
        
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if background is not None:
                # Resize background
                bg_resized = cv2.resize(background, self.resolution[::-1])
                bg_rgb = cv2.cvtColor(bg_resized, cv2.COLOR_BGR2RGB)
                ax.imshow(bg_rgb, alpha=0.5)
            
            im = ax.imshow(heatmap, cmap=self.colormap, alpha=0.7)
            plt.colorbar(im, ax=ax, label='Activity Density')
            ax.set_title('Movement Heatmap')
            ax.axis('off')
            
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Heatmap saved: {output_path}")
        else:
            # Fallback: save as raw image
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            cv2.imwrite(output_path, cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET))
    
    def reset(self):
        """Reset heatmap accumulator."""
        self.heatmap = np.zeros(self.resolution, dtype=np.float32)


class BirdsEyeView:
    """
    Project player positions to bird's-eye view using homography.
    """
    
    # Standard pitch dimensions (in meters)
    FOOTBALL_PITCH = (105, 68)
    BASKETBALL_COURT = (28, 15)
    CRICKET_PITCH = (20, 3)
    
    def __init__(
        self,
        pitch_width: float = 105,
        pitch_height: float = 68,
        output_size: Tuple[int, int] = (1050, 680),
        pitch_color: Tuple[int, int, int] = (34, 139, 34)
    ):
        """
        Initialize bird's-eye view projector.
        
        Args:
            pitch_width: Pitch width in meters
            pitch_height: Pitch height in meters
            output_size: Output image size (pixels)
            pitch_color: Background color (BGR)
        """
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.output_size = output_size
        self.pitch_color = pitch_color
        
        # Homography matrix
        self.homography: Optional[np.ndarray] = None
        
        # Source points (video frame corners)
        self.src_points: Optional[np.ndarray] = None
        
        # Destination points (pitch corners)
        self.dst_points = np.array([
            [0, 0],
            [output_size[0], 0],
            [output_size[0], output_size[1]],
            [0, output_size[1]]
        ], dtype=np.float32)
        
        # Pixels per meter
        self.pixels_per_meter = output_size[0] / pitch_width
    
    def set_source_points(self, points: List[Tuple[int, int]]):
        """
        Set source points (4 corners in video frame).
        
        Args:
            points: List of 4 (x, y) coordinates in order:
                    [top-left, top-right, bottom-right, bottom-left]
        """
        if len(points) != 4:
            raise ValueError("Exactly 4 source points required")
        
        self.src_points = np.array(points, dtype=np.float32)
        self.homography = cv2.getPerspectiveTransform(
            self.src_points, self.dst_points
        )
        logger.info("Homography matrix computed")
    
    def auto_detect_corners(self, frame: np.ndarray) -> bool:
        """
        Attempt to auto-detect court corners.
        
        Args:
            frame: Video frame
            
        Returns:
            True if corners were detected
        """
        # This is a simplified version - real implementation would use
        # line detection, corner detection, and template matching
        
        from .preprocessor import detect_court_lines, find_corner_candidates
        
        lines = detect_court_lines(frame)
        if len(lines) < 4:
            logger.warning("Not enough lines detected for auto corner detection")
            return False
        
        h, w = frame.shape[:2]
        corners = find_corner_candidates(lines, (h, w))
        
        if len(corners) < 4:
            logger.warning("Not enough corner candidates found")
            return False
        
        # Select 4 corners that form a quadrilateral
        # Simplified: use convex hull
        corners_array = np.array(corners)
        hull = cv2.convexHull(corners_array)
        
        if len(hull) >= 4:
            # Take 4 most separated points
            hull_points = hull.reshape(-1, 2)
            # Sort by angle from center
            center = hull_points.mean(axis=0)
            angles = np.arctan2(hull_points[:, 1] - center[1], 
                              hull_points[:, 0] - center[0])
            sorted_idx = np.argsort(angles)
            
            if len(sorted_idx) >= 4:
                selected = hull_points[sorted_idx[:4]]
                self.set_source_points(selected.tolist())
                return True
        
        return False
    
    def project_point(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """
        Project a point from video coordinates to bird's-eye view.
        
        Args:
            x, y: Point in video frame coordinates
            
        Returns:
            Projected point or None if no homography
        """
        if self.homography is None:
            return None
        
        point = np.array([[[x, y]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(point, self.homography)
        
        return (projected[0, 0, 0], projected[0, 0, 1])
    
    def project_points(
        self,
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Project multiple points."""
        if self.homography is None:
            return []
        
        points_array = np.array([[p] for p in points], dtype=np.float32)
        projected = cv2.perspectiveTransform(points_array, self.homography)
        
        return [(p[0][0], p[0][1]) for p in projected]
    
    def create_pitch_image(self) -> np.ndarray:
        """Create blank pitch background image."""
        img = np.full(
            (self.output_size[1], self.output_size[0], 3),
            self.pitch_color,
            dtype=np.uint8
        )
        
        # Draw pitch lines (white)
        line_color = (255, 255, 255)
        thickness = 2
        
        w, h = self.output_size
        
        # Outer boundary
        cv2.rectangle(img, (0, 0), (w-1, h-1), line_color, thickness)
        
        # Center line
        cv2.line(img, (w//2, 0), (w//2, h), line_color, thickness)
        
        # Center circle
        cv2.circle(img, (w//2, h//2), int(9.15 * self.pixels_per_meter), line_color, thickness)
        
        # Penalty areas (football)
        pa_width = int(40.3 * self.pixels_per_meter)
        pa_height = int(16.5 * self.pixels_per_meter)
        
        # Left penalty area
        cv2.rectangle(img, (0, (h-pa_width)//2), (pa_height, (h+pa_width)//2), line_color, thickness)
        
        # Right penalty area
        cv2.rectangle(img, (w-pa_height, (h-pa_width)//2), (w, (h+pa_width)//2), line_color, thickness)
        
        return img
    
    def render(
        self,
        positions: List[Tuple[float, float]],
        track_ids: List[int],
        team_assignments: Optional[Dict[int, int]] = None
    ) -> np.ndarray:
        """
        Render bird's-eye view with player positions.
        
        Args:
            positions: List of (x, y) positions in video coordinates
            track_ids: List of track IDs
            team_assignments: Optional team assignments for coloring
            
        Returns:
            Bird's-eye view image
        """
        img = self.create_pitch_image()
        
        if self.homography is None:
            return img
        
        # Project positions
        projected = self.project_points(positions)
        
        # Team colors
        team_colors = {
            0: (255, 0, 0),    # Blue
            1: (0, 0, 255),    # Red
            2: (0, 255, 0),    # Green
        }
        
        # Draw players
        for (px, py), track_id in zip(projected, track_ids):
            # Clamp to bounds
            px = max(0, min(self.output_size[0]-1, int(px)))
            py = max(0, min(self.output_size[1]-1, int(py)))
            
            # Get color
            if team_assignments and track_id in team_assignments:
                color = team_colors.get(team_assignments[track_id], (255, 255, 255))
            else:
                color = (255, 255, 255)
            
            # Draw player dot
            cv2.circle(img, (px, py), 8, color, -1)
            cv2.circle(img, (px, py), 8, (0, 0, 0), 1)
            
            # Draw ID
            cv2.putText(
                img, str(track_id), (px-5, py-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
        
        return img


class TeamClusterer:
    """
    Cluster players into teams based on jersey color.
    """
    
    def __init__(
        self,
        num_teams: int = 2,
        color_space: str = "hsv",
        use_histogram: bool = True
    ):
        """
        Initialize team clusterer.
        
        Args:
            num_teams: Number of teams to identify
            color_space: Color space for feature extraction
            use_histogram: Use color histogram features
        """
        self.num_teams = num_teams
        self.color_space = color_space
        self.use_histogram = use_histogram
        
        # Feature storage
        self.track_features: Dict[int, List[np.ndarray]] = defaultdict(list)
        
        # Team assignments
        self.assignments: Dict[int, int] = {}
        
        # KMeans model
        self.kmeans = None
    
    def extract_color_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract color features from player crop.
        
        Args:
            crop: Player crop image (BGR)
            
        Returns:
            Feature vector
        """
        if crop.size == 0:
            return np.zeros(24)  # Default feature size
        
        # Focus on torso region (middle third)
        h, w = crop.shape[:2]
        torso = crop[h//3:2*h//3, w//4:3*w//4]
        
        if torso.size == 0:
            torso = crop
        
        # Convert color space
        if self.color_space == "hsv":
            converted = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        elif self.color_space == "lab":
            converted = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
        else:
            converted = torso
        
        if self.use_histogram:
            # Color histogram
            hist = cv2.calcHist(
                [converted], [0, 1], None,
                [8, 8], [0, 256, 0, 256]
            )
            features = cv2.normalize(hist, hist).flatten()
        else:
            # Mean color
            features = converted.reshape(-1, 3).mean(axis=0)
        
        return features
    
    def add_observation(self, track_id: int, crop: np.ndarray):
        """Add color observation for a track."""
        features = self.extract_color_features(crop)
        self.track_features[track_id].append(features)
        
        # Keep only recent observations
        if len(self.track_features[track_id]) > 30:
            self.track_features[track_id] = self.track_features[track_id][-30:]
    
    def cluster(self) -> Dict[int, int]:
        """
        Perform clustering and assign teams.
        
        Returns:
            Dict mapping track_id to team_id
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, team clustering disabled")
            return {}
        
        if len(self.track_features) < self.num_teams:
            return {}
        
        # Aggregate features per track
        track_ids = []
        features = []
        
        for track_id, feat_list in self.track_features.items():
            if len(feat_list) >= 5:  # Minimum observations
                track_ids.append(track_id)
                # Use mean of recent features
                mean_feat = np.mean(feat_list[-10:], axis=0)
                features.append(mean_feat)
        
        if len(features) < self.num_teams:
            return {}
        
        features = np.array(features)
        
        # Run KMeans
        self.kmeans = KMeans(n_clusters=self.num_teams, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(features)
        
        # Update assignments
        for track_id, label in zip(track_ids, labels):
            self.assignments[track_id] = int(label)
        
        logger.info(f"Clustered {len(self.assignments)} tracks into {self.num_teams} teams")
        
        return self.assignments
    
    def get_team(self, track_id: int) -> Optional[int]:
        """Get team assignment for a track."""
        return self.assignments.get(track_id)


class SpeedEstimator:
    """
    Estimate player speeds from track positions.
    """
    
    def __init__(
        self,
        fps: float,
        pixels_per_meter: Optional[float] = None,
        smoothing_window: int = 5
    ):
        """
        Initialize speed estimator.
        
        Args:
            fps: Video frame rate
            pixels_per_meter: Calibration factor
            smoothing_window: Window for speed smoothing
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter or 10.0  # Default estimate
        self.smoothing_window = smoothing_window
        
        # Track positions
        self.positions: Dict[int, List[Tuple[float, float, int]]] = defaultdict(list)
        
        # Speed history
        self.speeds: Dict[int, List[float]] = defaultdict(list)
    
    def set_calibration(self, pixels_per_meter: float):
        """Set calibration factor."""
        self.pixels_per_meter = pixels_per_meter
    
    def update(
        self,
        track_id: int,
        x: float,
        y: float,
        frame_id: int
    ) -> Optional[float]:
        """
        Update position and calculate speed.
        
        Args:
            track_id: Track ID
            x, y: Current position
            frame_id: Current frame number
            
        Returns:
            Current speed in m/s or None
        """
        self.positions[track_id].append((x, y, frame_id))
        
        # Need at least 2 positions
        if len(self.positions[track_id]) < 2:
            return None
        
        # Calculate displacement
        prev = self.positions[track_id][-2]
        curr = self.positions[track_id][-1]
        
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        dist_pixels = np.sqrt(dx**2 + dy**2)
        
        # Convert to meters
        dist_meters = dist_pixels / self.pixels_per_meter
        
        # Time elapsed (frames)
        dt_frames = curr[2] - prev[2]
        dt_seconds = dt_frames / self.fps
        
        if dt_seconds <= 0:
            return None
        
        # Speed in m/s
        speed = dist_meters / dt_seconds
        
        self.speeds[track_id].append(speed)
        
        # Smooth speed
        if len(self.speeds[track_id]) >= self.smoothing_window:
            smoothed = np.mean(self.speeds[track_id][-self.smoothing_window:])
            return smoothed
        
        return speed
    
    def get_stats(self, track_id: int) -> Dict[str, float]:
        """Get speed statistics for a track."""
        speeds = self.speeds.get(track_id, [])
        
        if not speeds:
            return {'avg': 0, 'max': 0, 'min': 0}
        
        return {
            'avg': float(np.mean(speeds)),
            'max': float(np.max(speeds)),
            'min': float(np.min(speeds))
        }
    
    def get_all_stats(self) -> Dict[int, Dict[str, float]]:
        """Get speed statistics for all tracks."""
        return {tid: self.get_stats(tid) for tid in self.speeds}


class MetricsCalculator:
    """
    Calculate evaluation metrics for tracking quality.
    """
    
    def __init__(self):
        self.detection_counts: List[int] = []
        self.track_lengths: Dict[int, int] = defaultdict(int)
        self.track_starts: Dict[int, int] = {}
        self.track_ends: Dict[int, int] = {}
        self.id_switches: int = 0
        self.fragmentations: int = 0
    
    def update(
        self,
        frame_id: int,
        track_ids: List[int],
        num_detections: int
    ):
        """Update metrics with frame data."""
        self.detection_counts.append(num_detections)
        
        for track_id in track_ids:
            self.track_lengths[track_id] += 1
            
            if track_id not in self.track_starts:
                self.track_starts[track_id] = frame_id
            
            self.track_ends[track_id] = frame_id
    
    def record_id_switch(self):
        """Record an ID switch event."""
        self.id_switches += 1
    
    def record_fragmentation(self):
        """Record a track fragmentation event."""
        self.fragmentations += 1
    
    def calculate(self) -> Dict[str, Any]:
        """
        Calculate all metrics.
        
        Returns:
            Dict of metric values
        """
        total_tracks = len(self.track_lengths)
        
        if total_tracks == 0:
            return {
                'total_tracks': 0,
                'avg_track_length': 0,
                'total_detections': 0,
                'avg_detections_per_frame': 0,
                'id_switches': 0,
                'fragmentations': 0,
                'fragmentation_rate': 0
            }
        
        track_lengths = list(self.track_lengths.values())
        
        return {
            'total_tracks': total_tracks,
            'avg_track_length': float(np.mean(track_lengths)),
            'min_track_length': int(np.min(track_lengths)),
            'max_track_length': int(np.max(track_lengths)),
            'total_detections': sum(self.detection_counts),
            'avg_detections_per_frame': float(np.mean(self.detection_counts)),
            'id_switches': self.id_switches,
            'fragmentations': self.fragmentations,
            'fragmentation_rate': self.fragmentations / total_tracks if total_tracks > 0 else 0
        }
    
    def save(self, output_path: str):
        """Save metrics to JSON file."""
        metrics = self.calculate()
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved: {output_path}")


class AnalyticsEngine:
    """
    Main analytics engine combining all enhancement modules.
    """
    
    def __init__(
        self,
        enable_heatmap: bool = True,
        enable_birds_eye: bool = True,
        enable_team_clustering: bool = True,
        enable_speed: bool = True,
        enable_metrics: bool = True,
        fps: float = 30.0,
        frame_size: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize analytics engine.
        
        Args:
            enable_heatmap: Enable heatmap generation
            enable_birds_eye: Enable bird's-eye view
            enable_team_clustering: Enable team clustering
            enable_speed: Enable speed estimation
            enable_metrics: Enable metrics calculation
            fps: Video frame rate
            frame_size: (width, height) of frames
        """
        self.fps = fps
        self.frame_size = frame_size
        
        # Initialize components
        self.heatmap = HeatmapGenerator() if enable_heatmap else None
        self.birds_eye = BirdsEyeView() if enable_birds_eye else None
        self.team_clusterer = TeamClusterer() if enable_team_clustering else None
        self.speed_estimator = SpeedEstimator(fps=fps) if enable_speed else None
        self.metrics = MetricsCalculator() if enable_metrics else None
        
        # Count over time data
        self.count_history: List[Tuple[int, int]] = []
        
        logger.success("AnalyticsEngine initialized")
    
    def set_frame_size(self, width: int, height: int):
        """Set frame size for all components."""
        self.frame_size = (width, height)
        
        if self.heatmap:
            self.heatmap.set_frame_size(width, height)
    
    def set_homography_points(self, points: List[Tuple[int, int]]):
        """Set homography source points for bird's-eye view."""
        if self.birds_eye:
            self.birds_eye.set_source_points(points)
            
            if self.speed_estimator:
                self.speed_estimator.set_calibration(
                    self.birds_eye.pixels_per_meter
                )
    
    def update(
        self,
        frame: np.ndarray,
        frame_id: int,
        boxes: np.ndarray,
        track_ids: List[int],
        num_detections: int
    ):
        """
        Update all analytics with frame data.
        
        Args:
            frame: Current frame
            frame_id: Frame number
            boxes: Detection boxes (N, 4)
            track_ids: Track IDs
            num_detections: Number of detections
        """
        # Update frame size if needed
        if self.frame_size is None:
            h, w = frame.shape[:2]
            self.set_frame_size(w, h)
        
        # Count history
        self.count_history.append((frame_id, len(track_ids)))
        
        # Process each track
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Heatmap
            if self.heatmap:
                self.heatmap.add_position(cx, cy)
            
            # Speed
            if self.speed_estimator:
                self.speed_estimator.update(track_id, cx, cy, frame_id)
            
            # Team clustering
            if self.team_clusterer:
                # Extract crop for color analysis
                x1i, y1i = max(0, int(x1)), max(0, int(y1))
                x2i, y2i = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                
                if x2i > x1i and y2i > y1i:
                    crop = frame[y1i:y2i, x1i:x2i]
                    self.team_clusterer.add_observation(track_id, crop)
        
        # Metrics
        if self.metrics:
            self.metrics.update(frame_id, track_ids, num_detections)
    
    def get_team_assignments(self) -> Dict[int, int]:
        """Get team assignments from clustering."""
        if self.team_clusterer:
            return self.team_clusterer.cluster()
        return {}
    
    def get_heatmap_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Get heatmap overlay on frame."""
        if self.heatmap:
            return self.heatmap.get_overlay(frame)
        return frame
    
    def get_birds_eye_view(
        self,
        positions: List[Tuple[float, float]],
        track_ids: List[int]
    ) -> np.ndarray:
        """Get bird's-eye view rendering."""
        if self.birds_eye:
            team_assignments = self.get_team_assignments()
            return self.birds_eye.render(positions, track_ids, team_assignments)
        return np.zeros((680, 1050, 3), dtype=np.uint8)
    
    def get_speed_stats(self) -> Dict[int, Dict[str, float]]:
        """Get speed statistics for all tracks."""
        if self.speed_estimator:
            return self.speed_estimator.get_all_stats()
        return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get evaluation metrics."""
        if self.metrics:
            return self.metrics.calculate()
        return {}
    
    def save_count_plot(self, output_path: str):
        """Save object count over time plot."""
        if not MATPLOTLIB_AVAILABLE or not self.count_history:
            return
        
        frames, counts = zip(*self.count_history)
        
        plt.figure(figsize=(12, 4))
        plt.plot(frames, counts, 'b-', linewidth=1)
        plt.fill_between(frames, counts, alpha=0.3)
        plt.xlabel('Frame')
        plt.ylabel('Number of Tracked Objects')
        plt.title('Object Count Over Time')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Count plot saved: {output_path}")
    
    def save_all_outputs(self, output_dir: str, background_frame: Optional[np.ndarray] = None):
        """Save all analytics outputs."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Heatmap
        if self.heatmap:
            self.heatmap.save_heatmap(
                str(output_path / "heatmap.png"),
                background=background_frame
            )
        
        # Count plot
        self.save_count_plot(str(output_path / "count_plot.png"))
        
        # Metrics
        if self.metrics:
            self.metrics.save(str(output_path / "metrics.json"))
        
        # Speed stats
        if self.speed_estimator:
            speed_stats = self.get_speed_stats()
            with open(output_path / "speed_stats.json", 'w') as f:
                json.dump(speed_stats, f, indent=2)
        
        logger.success(f"All analytics saved to: {output_dir}")
