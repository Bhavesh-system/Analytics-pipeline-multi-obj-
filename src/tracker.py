"""
Tracker Module - Kalman Filter Multi-Object Tracking
=====================================================

This module implements multi-object tracking using Kalman filtering
and Hungarian algorithm for data association. No external tracking
libraries required - pure numpy/scipy implementation.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import cv2
from loguru import logger
from scipy.optimize import linear_sum_assignment

# Kalman filter tracker - no external dependencies needed
TRACKER_AVAILABLE = True
logger.success("Kalman filter tracker loaded")


@dataclass
class Track:
    """Single track with persistent ID."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int = 0
    age: int = 0  # Frames since track started
    hits: int = 1  # Number of detections associated
    time_since_update: int = 0
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class FrameTracks:
    """All tracks for a single frame."""
    frame_id: int
    tracks: List[Track]
    
    @property
    def num_tracks(self) -> int:
        return len(self.tracks)
    
    def get_track_ids(self) -> List[int]:
        return [t.track_id for t in self.tracks]
    
    def get_boxes(self) -> np.ndarray:
        if not self.tracks:
            return np.empty((0, 4))
        return np.array([t.bbox for t in self.tracks])
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None


@dataclass
class TrackHistory:
    """Track history for trajectory visualization."""
    track_id: int
    positions: List[Tuple[float, float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)
    
    def add_position(self, x: float, y: float, conf: float, frame_id: int):
        self.positions.append((x, y))
        self.confidences.append(conf)
        self.frame_ids.append(frame_id)
    
    def get_recent_positions(self, n: int = 30) -> List[Tuple[float, float]]:
        return self.positions[-n:]


class KalmanBoxTracker:
    """
    Kalman Filter tracker for a single bounding box.
    State: [x_center, y_center, area, aspect_ratio, vx, vy, va]
    """
    count = 0
    
    def __init__(self, bbox: np.ndarray):
        """Initialize tracker with bounding box [x1, y1, x2, y2]."""
        # State: [x, y, s, r, vx, vy, vs] where s=area, r=aspect_ratio
        self.kf = self._create_kalman_filter()
        
        # Convert bbox to state
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        s = w * h
        r = w / max(h, 1e-6)
        
        self.kf['x'][:4] = np.array([x, y, s, r]).reshape(-1, 1)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.confidence = 0.5
    
    def _create_kalman_filter(self) -> Dict:
        """Create Kalman filter matrices."""
        # State transition matrix
        F = np.eye(7)
        F[0, 4] = 1  # x += vx
        F[1, 5] = 1  # y += vy
        F[2, 6] = 1  # s += vs
        
        # Measurement matrix (we observe x, y, s, r)
        H = np.zeros((4, 7))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # s
        H[3, 3] = 1  # r
        
        # Process noise
        Q = np.eye(7)
        Q[4:, 4:] *= 0.01
        
        # Measurement noise
        R = np.eye(4)
        R[2, 2] *= 10
        R[3, 3] *= 10
        
        # Initial covariance
        P = np.eye(7) * 10
        P[4:, 4:] *= 1000
        
        # State vector
        x = np.zeros((7, 1))
        
        return {'F': F, 'H': H, 'Q': Q, 'R': R, 'P': P, 'x': x}
    
    def update(self, bbox: np.ndarray, confidence: float = 0.5):
        """Update state with observed bbox."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.confidence = confidence
        
        # Convert bbox to measurement
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        s = w * h
        r = w / max(h, 1e-6)
        z = np.array([x, y, s, r]).reshape(-1, 1)
        
        # Kalman update
        kf = self.kf
        y_res = z - kf['H'] @ kf['x']
        S = kf['H'] @ kf['P'] @ kf['H'].T + kf['R']
        K = kf['P'] @ kf['H'].T @ np.linalg.inv(S)
        kf['x'] = kf['x'] + K @ y_res
        kf['P'] = (np.eye(7) - K @ kf['H']) @ kf['P']
    
    def predict(self) -> np.ndarray:
        """Predict next state and return bbox."""
        kf = self.kf
        
        # Handle negative area
        if kf['x'][6] + kf['x'][2] <= 0:
            kf['x'][6] *= 0.0
        
        # Kalman predict
        kf['x'] = kf['F'] @ kf['x']
        kf['P'] = kf['F'] @ kf['P'] @ kf['F'].T + kf['Q']
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        self.history.append(self.get_state())
        return self.history[-1]
    
    def get_state(self) -> np.ndarray:
        """Get current state as bbox [x1, y1, x2, y2]."""
        x, y, s, r = self.kf['x'][:4].flatten()
        w = np.sqrt(max(s * r, 0))
        h = max(s / max(w, 1e-6), 0)
        return np.array([x - w/2, y - h/2, x + w/2, y + h/2])


class KalmanTracker:
    """
    Multi-object tracker using Kalman filtering and Hungarian algorithm.
    Pure numpy/scipy implementation - no external tracking libraries needed.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum frames to keep track alive without detection
            min_hits: Minimum hits before track is confirmed
            iou_threshold: Minimum IoU for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        logger.info(f"Kalman tracker initialized (max_age={max_age}, min_hits={min_hits})")
    
    def _iou_batch(self, bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
        """Compute IoU between two sets of boxes."""
        if len(bb_test) == 0 or len(bb_gt) == 0:
            return np.empty((len(bb_test), len(bb_gt)))
        
        xx1 = np.maximum(bb_test[:, None, 0], bb_gt[None, :, 0])
        yy1 = np.maximum(bb_test[:, None, 1], bb_gt[None, :, 1])
        xx2 = np.minimum(bb_test[:, None, 2], bb_gt[None, :, 2])
        yy2 = np.minimum(bb_test[:, None, 3], bb_gt[None, :, 3])
        
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        inter = w * h
        
        area_test = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
        area_gt = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])
        
        union = area_test[:, None] + area_gt[None, :] - inter
        return inter / np.maximum(union, 1e-6)
    
    def _associate_detections_to_trackers(
        self,
        detections: np.ndarray,
        trackers: np.ndarray
    ) -> Tuple[List, List, List]:
        """
        Associate detections to tracked objects using Hungarian algorithm.
        
        Returns:
            matches: List of (det_idx, trk_idx) tuples
            unmatched_detections: List of detection indices
            unmatched_trackers: List of tracker indices
        """
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(trackers)))
        
        iou_matrix = self._iou_batch(detections, trackers)
        
        # Hungarian algorithm
        if min(iou_matrix.shape) > 0:
            # Use linear_sum_assignment (minimization, so use negative IoU)
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.column_stack((row_indices, col_indices))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
        
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filter out low IoU matches
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.tolist())
        
        return matches, unmatched_detections, unmatched_trackers
    
    def update(
        self,
        detections: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        frame: np.ndarray
    ) -> np.ndarray:
        """
        Update tracker with new detections.
        
        Args:
            detections: (N, 4) array of [x1, y1, x2, y2]
            confidences: (N,) array of confidence scores
            class_ids: (N,) array of class IDs (unused but kept for API)
            frame: Current frame (unused but kept for API)
            
        Returns:
            (M, 6) array of [x1, y1, x2, y2, track_id, conf]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # Remove bad trackers
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = np.delete(trks, to_del, axis=0) if to_del else trks
        
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = \
            self._associate_detections_to_trackers(detections, trks)
        
        # Update matched trackers
        for m in matched:
            det_idx, trk_idx = m
            self.trackers[trk_idx].update(detections[det_idx], confidences[det_idx])
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i])
            trk.confidence = confidences[i]
            self.trackers.append(trk)
        
        # Build output
        results = []
        for trk in self.trackers:
            # Only return confirmed tracks
            if trk.time_since_update < 1 and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                d = trk.get_state()
                results.append([d[0], d[1], d[2], d[3], trk.id, trk.confidence])
        
        # Remove dead tracks
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        return np.array(results) if results else np.empty((0, 6))


class SimpleTracker:
    """Simple IoU-based tracker fallback when norfair is not available."""
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1
        
    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def update(
        self,
        detections: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        frame: np.ndarray
    ) -> np.ndarray:
        """Update tracker with simple IoU matching."""
        # Age all tracks
        for tid in list(self.tracks.keys()):
            self.tracks[tid]['age'] += 1
            if self.tracks[tid]['age'] > self.max_age:
                del self.tracks[tid]
        
        if len(detections) == 0:
            return np.empty((0, 6))
        
        results = []
        matched_tracks = set()
        matched_dets = set()
        
        # Match detections to existing tracks
        for det_idx, (det, conf) in enumerate(zip(detections, confidences)):
            best_iou = self.iou_threshold
            best_tid = None
            
            for tid, track in self.tracks.items():
                if tid in matched_tracks:
                    continue
                iou = self._iou(det, track['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            
            if best_tid is not None:
                self.tracks[best_tid] = {'box': det, 'age': 0, 'conf': conf}
                results.append([*det, best_tid, conf])
                matched_tracks.add(best_tid)
                matched_dets.add(det_idx)
        
        # Create new tracks for unmatched detections
        for det_idx, (det, conf) in enumerate(zip(detections, confidences)):
            if det_idx not in matched_dets:
                self.tracks[self.next_id] = {'box': det, 'age': 0, 'conf': conf}
                results.append([*det, self.next_id, conf])
                self.next_id += 1
        
        return np.array(results) if results else np.empty((0, 6))


class MultiTracker:
    """
    Multi-object tracker using Kalman filtering and Hungarian algorithm.
    
    Provides robust tracking with automatic track management,
    occlusion handling, and trajectory prediction.
    """
    
    def __init__(
        self,
        primary_tracker: str = "kalman",
        secondary_tracker: str = None,
        ensemble_enabled: bool = False,
        iou_merge_threshold: float = 0.5,
        bytetrack_config: Optional[Dict] = None,
        botsort_config: Optional[Dict] = None,
        frame_rate: int = 30
    ):
        """
        Initialize multi-tracker.
        
        Args:
            primary_tracker: Primary tracking algorithm (kalman or simple)
            secondary_tracker: Not used (kept for compatibility)
            ensemble_enabled: Not used (kept for compatibility)
            iou_merge_threshold: IoU threshold for merging tracks
            bytetrack_config: Kalman tracker configuration
            botsort_config: Not used
            frame_rate: Video frame rate
        """
        self.ensemble_enabled = False  # Ensemble not needed with Kalman
        self.iou_merge_threshold = iou_merge_threshold
        self.frame_rate = frame_rate
        
        # Initialize tracker
        self.primary = None
        self.secondary = None
        
        if TRACKER_AVAILABLE:
            try:
                kalman_config = {
                    'max_age': 30,
                    'min_hits': 3,
                    'iou_threshold': 0.3
                }
                if bytetrack_config:
                    # Map old config names to new ones
                    if 'track_buffer' in bytetrack_config:
                        kalman_config['max_age'] = bytetrack_config['track_buffer']
                    kalman_config.update({k: v for k, v in bytetrack_config.items() 
                                         if k in kalman_config})
                
                self.primary = KalmanTracker(**kalman_config)
            except Exception as e:
                logger.warning(f"Failed to initialize Kalman tracker: {e}")
                self.primary = SimpleTracker()
        else:
            logger.warning("Using simple IoU tracker")
            self.primary = SimpleTracker()
        
        # Track history for visualization
        self.track_histories: Dict[int, TrackHistory] = {}
        
        # ID mapping for ensemble (kept for compatibility)
        self.id_mapping: Dict[int, int] = {}
        self.next_unified_id = 1
        
        # Fragmentation logging
        self.fragmentation_log: List[Dict] = []
        
        logger.success("MultiTracker initialized")
    
    def _compute_iou_matrix(
        self,
        boxes1: np.ndarray,
        boxes2: np.ndarray
    ) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes."""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.empty((len(boxes1), len(boxes2)))
        
        # Compute intersections
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
        
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Compute areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        union = area1[:, None] + area2[None, :] - inter
        
        return inter / np.maximum(union, 1e-6)
    
    def _merge_tracks(
        self,
        tracks1: np.ndarray,
        tracks2: np.ndarray
    ) -> np.ndarray:
        """Merge tracks from two trackers using IoU matching."""
        if len(tracks1) == 0:
            return tracks2
        if len(tracks2) == 0:
            return tracks1
        
        # Compute IoU matrix
        boxes1 = tracks1[:, :4]
        boxes2 = tracks2[:, :4]
        iou_matrix = self._compute_iou_matrix(boxes1, boxes2)
        
        # Merge tracks with high IoU
        merged = []
        used2 = set()
        
        for i, track1 in enumerate(tracks1):
            best_match = -1
            best_iou = self.iou_merge_threshold
            
            for j, iou in enumerate(iou_matrix[i]):
                if j not in used2 and iou > best_iou:
                    best_iou = iou
                    best_match = j
            
            if best_match >= 0:
                # Average the boxes, keep track1's ID
                track2 = tracks2[best_match]
                avg_box = (track1[:4] + track2[:4]) / 2
                max_conf = max(track1[5], track2[5])
                merged.append([*avg_box, track1[4], max_conf])
                used2.add(best_match)
            else:
                merged.append(track1)
        
        # Add unmatched tracks from tracker2
        for j, track2 in enumerate(tracks2):
            if j not in used2:
                merged.append(track2)
        
        return np.array(merged) if merged else np.empty((0, 6))
    
    def update(
        self,
        detections: np.ndarray,
        confidences: np.ndarray,
        frame: np.ndarray,
        frame_id: int = 0,
        class_ids: Optional[np.ndarray] = None
    ) -> FrameTracks:
        """
        Update tracker with new detections.
        
        Args:
            detections: (N, 4) array of [x1, y1, x2, y2]
            confidences: (N,) array of confidence scores
            frame: Current frame
            frame_id: Frame number
            class_ids: Optional class IDs
            
        Returns:
            FrameTracks with all active tracks
        """
        if class_ids is None:
            class_ids = np.zeros(len(detections), dtype=int)
        
        # Run primary tracker
        tracks1 = self.primary.update(detections, confidences, class_ids, frame)
        
        # Run secondary tracker if enabled
        if self.ensemble_enabled and self.secondary is not None:
            tracks2 = self.secondary.update(detections, confidences, class_ids, frame)
            raw_tracks = self._merge_tracks(tracks1, tracks2)
        else:
            raw_tracks = tracks1
        
        # Convert to Track objects
        tracks = []
        for track_data in raw_tracks:
            if len(track_data) >= 6:
                bbox = track_data[:4]
                track_id = int(track_data[4])
                conf = float(track_data[5])
            else:
                continue
            
            track = Track(
                track_id=track_id,
                bbox=bbox,
                confidence=conf,
                class_id=0
            )
            tracks.append(track)
            
            # Update history
            if track_id not in self.track_histories:
                self.track_histories[track_id] = TrackHistory(track_id=track_id)
            
            cx, cy = track.center
            self.track_histories[track_id].add_position(cx, cy, conf, frame_id)
        
        return FrameTracks(frame_id=frame_id, tracks=tracks)
    
    def get_track_history(self, track_id: int) -> Optional[TrackHistory]:
        """Get history for a specific track."""
        return self.track_histories.get(track_id)
    
    def get_all_histories(self) -> Dict[int, TrackHistory]:
        """Get all track histories."""
        return self.track_histories
    
    def log_fragmentation(self, old_id: int, new_id: int, frame_id: int, reason: str):
        """Log track fragmentation event."""
        self.fragmentation_log.append({
            'frame_id': frame_id,
            'old_id': old_id,
            'new_id': new_id,
            'reason': reason
        })
    
    def get_fragmentation_log(self) -> List[Dict]:
        """Get fragmentation log for analysis."""
        return self.fragmentation_log
    
    def get_tracker_info(self) -> Dict[str, Any]:
        """Get tracker configuration for reporting."""
        return {
            "primary_tracker": type(self.primary).__name__,
            "secondary_tracker": type(self.secondary).__name__ if self.secondary else None,
            "ensemble_enabled": self.ensemble_enabled,
            "iou_merge_threshold": self.iou_merge_threshold,
            "total_tracks": len(self.track_histories),
            "fragmentation_events": len(self.fragmentation_log)
        }
