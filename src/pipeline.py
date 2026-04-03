"""
Pipeline Module - Main Orchestrator
====================================

This module ties together all components into a unified pipeline:
Preprocessor → Detector → Tracker → ReID → Annotator → Analytics
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
import yaml
import json
import numpy as np
import cv2
from loguru import logger
from dataclasses import dataclass

from .preprocessor import VideoPreprocessor
from .detector import Detector, FrameDetections
from .tracker import MultiTracker, FrameTracks
from .reid import ReIDModule
from .annotator import VideoAnnotator, VideoWriter, AnnotationConfig
from .analytics import AnalyticsEngine
from .report import ReportGenerator


@dataclass
class PipelineConfig:
    """Pipeline configuration with optimized defaults."""
    # Paths
    output_dir: str = "outputs"
    temp_dir: str = "temp"
    
    # Device
    device: str = "cuda"
    
    # Preprocessing
    frame_skip: int = 2  # Process every 2nd frame for speed
    max_dimension: int = 960  # Smaller for speed
    target_fps: Optional[float] = None
    
    # Detection
    detection_model: str = "yolov8m.pt"  # Medium model - balanced
    confidence_threshold: float = 0.35
    nms_iou_threshold: float = 0.45
    imgsz: int = 640  # Smaller input for speed
    
    # Tracking
    primary_tracker: str = "kalman"
    secondary_tracker: str = None
    ensemble_enabled: bool = False
    
    # ReID
    reid_enabled: bool = True
    reid_model: str = "mobilenet_v3_small"
    similarity_threshold: float = 0.60
    
    # Annotation
    trajectory_length: int = 30
    show_confidence: bool = True
    color_by_team: bool = True
    
    # Analytics
    enable_heatmap: bool = True
    enable_birds_eye: bool = True
    enable_team_clustering: bool = True
    enable_speed: bool = True
    enable_metrics: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        # Map YAML sections to config attributes
        if 'general' in data:
            config.output_dir = data['general'].get('output_dir', config.output_dir)
            config.temp_dir = data['general'].get('temp_dir', config.temp_dir)
            config.device = data['general'].get('device', config.device)
        
        if 'preprocessing' in data:
            config.frame_skip = data['preprocessing'].get('frame_skip', config.frame_skip)
            config.max_dimension = data['preprocessing'].get('max_dimension', config.max_dimension)
            config.target_fps = data['preprocessing'].get('target_fps', config.target_fps)
        
        if 'detection' in data:
            config.detection_model = data['detection'].get('model', config.detection_model)
            config.confidence_threshold = data['detection'].get('confidence_threshold', config.confidence_threshold)
            config.nms_iou_threshold = data['detection'].get('nms_iou_threshold', config.nms_iou_threshold)
            config.imgsz = data['detection'].get('imgsz', config.imgsz)
        
        if 'tracking' in data:
            config.primary_tracker = data['tracking'].get('primary_tracker', config.primary_tracker)
            config.secondary_tracker = data['tracking'].get('secondary_tracker', config.secondary_tracker)
            config.ensemble_enabled = data['tracking'].get('ensemble', {}).get('enabled', config.ensemble_enabled)
        
        if 'reid' in data:
            config.reid_enabled = data['reid'].get('enabled', config.reid_enabled)
            config.reid_model = data['reid'].get('model', config.reid_model)
            config.similarity_threshold = data['reid'].get('similarity_threshold', config.similarity_threshold)
        
        if 'annotation' in data:
            config.trajectory_length = data['annotation'].get('trajectory_length', config.trajectory_length)
            config.show_confidence = data['annotation'].get('show_confidence', config.show_confidence)
            config.color_by_team = data['annotation'].get('color_by_team', config.color_by_team)
        
        if 'analytics' in data:
            config.enable_heatmap = data['analytics'].get('heatmap', {}).get('enabled', config.enable_heatmap)
            config.enable_birds_eye = data['analytics'].get('birds_eye', {}).get('enabled', config.enable_birds_eye)
            config.enable_team_clustering = data['analytics'].get('team_clustering', {}).get('enabled', config.enable_team_clustering)
            config.enable_speed = data['analytics'].get('speed_estimation', {}).get('enabled', config.enable_speed)
            config.enable_metrics = data['analytics'].get('metrics', {}).get('enabled', config.enable_metrics)
        
        return config


class TrackingPipeline:
    """
    Main tracking pipeline orchestrator.
    
    Coordinates all modules to process video and generate outputs.
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration object
            config_path: Path to config.yaml (alternative to config object)
        """
        # Load config
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = PipelineConfig.from_yaml(config_path)
        else:
            self.config = PipelineConfig()
        
        # Create output directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.temp_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components (lazy loading)
        self._preprocessor: Optional[VideoPreprocessor] = None
        self._detector: Optional[Detector] = None
        self._tracker: Optional[MultiTracker] = None
        self._reid: Optional[ReIDModule] = None
        self._annotator: Optional[VideoAnnotator] = None
        self._analytics: Optional[AnalyticsEngine] = None
        
        # Processing state
        self.video_info: Optional[Dict[str, Any]] = None
        self.processing_stats: Dict[str, Any] = {}
        
        logger.success("TrackingPipeline initialized")
    
    @property
    def preprocessor(self) -> VideoPreprocessor:
        """Lazy-load preprocessor."""
        if self._preprocessor is None:
            self._preprocessor = VideoPreprocessor(
                frame_skip=self.config.frame_skip,
                max_dimension=self.config.max_dimension,
                target_fps=self.config.target_fps,
                temp_dir=self.config.temp_dir
            )
        return self._preprocessor
    
    @property
    def detector(self) -> Detector:
        """Lazy-load detector."""
        if self._detector is None:
            self._detector = Detector(
                model_path=self.config.detection_model,
                confidence_threshold=self.config.confidence_threshold,
                nms_iou_threshold=self.config.nms_iou_threshold,
                device=self.config.device,
                imgsz=self.config.imgsz
            )
        return self._detector
    
    @property
    def tracker(self) -> MultiTracker:
        """Lazy-load tracker."""
        if self._tracker is None:
            self._tracker = MultiTracker(
                primary_tracker=self.config.primary_tracker,
                secondary_tracker=self.config.secondary_tracker,
                ensemble_enabled=self.config.ensemble_enabled,
                frame_rate=int(self.video_info.get('fps', 30)) if self.video_info else 30
            )
        return self._tracker
    
    @property
    def reid(self) -> Optional[ReIDModule]:
        """Lazy-load ReID module."""
        if self._reid is None and self.config.reid_enabled:
            self._reid = ReIDModule(
                model_name=self.config.reid_model,
                similarity_threshold=self.config.similarity_threshold,
                device=self.config.device
            )
        return self._reid
    
    @property
    def annotator(self) -> VideoAnnotator:
        """Lazy-load annotator."""
        if self._annotator is None:
            ann_config = AnnotationConfig(
                trajectory_length=self.config.trajectory_length,
                show_confidence=self.config.show_confidence,
                color_by_team=self.config.color_by_team
            )
            self._annotator = VideoAnnotator(config=ann_config)
        return self._annotator
    
    @property
    def analytics(self) -> AnalyticsEngine:
        """Lazy-load analytics engine."""
        if self._analytics is None:
            fps = self.video_info.get('fps', 30) if self.video_info else 30
            self._analytics = AnalyticsEngine(
                enable_heatmap=self.config.enable_heatmap,
                enable_birds_eye=self.config.enable_birds_eye,
                enable_team_clustering=self.config.enable_team_clustering,
                enable_speed=self.config.enable_speed,
                enable_metrics=self.config.enable_metrics,
                fps=fps
            )
        return self._analytics
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int
    ) -> Tuple[np.ndarray, FrameTracks, Dict[str, Any]]:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input frame (BGR)
            frame_id: Frame number
            
        Returns:
            Tuple of (annotated_frame, tracks, frame_stats)
        """
        start_time = time.time()
        
        # Detection
        detections = self.detector.detect(frame, frame_id)
        
        # Tracking
        tracks = self.tracker.update(
            detections.raw_boxes,
            detections.confidences,
            frame,
            frame_id
        )
        
        # ReID
        unified_ids = {}
        if self.reid and len(tracks.tracks) > 0:
            boxes = tracks.get_boxes()
            track_ids = tracks.get_track_ids()
            unified_ids = self.reid.update(frame, track_ids, boxes, frame_id)
        
        # Analytics update
        if len(tracks.tracks) > 0:
            boxes = tracks.get_boxes()
            track_ids = tracks.get_track_ids()
            self.analytics.update(
                frame, frame_id, boxes, track_ids, detections.num_detections
            )
        
        # Annotation
        if len(tracks.tracks) > 0:
            boxes = tracks.get_boxes()
            track_ids = tracks.get_track_ids()
            confs = np.array([t.confidence for t in tracks.tracks])
            
            annotated = self.annotator.annotate_frame(
                frame,
                boxes,
                track_ids,
                confidences=confs,
                unified_ids=unified_ids
            )
        else:
            annotated = frame.copy()
        
        # Add info overlay
        fps = 1.0 / (time.time() - start_time + 1e-6)
        annotated = self.annotator.add_info_overlay(
            annotated,
            frame_id,
            tracks.num_tracks,
            fps=fps
        )
        
        # Frame stats
        frame_stats = {
            'frame_id': frame_id,
            'num_detections': detections.num_detections,
            'num_tracks': tracks.num_tracks,
            'processing_time': time.time() - start_time
        }
        
        return annotated, tracks, frame_stats
    
    def run(
        self,
        input_source: str,
        output_name: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Run the full pipeline on a video.
        
        Args:
            input_source: Video URL or file path
            output_name: Output file name (without extension)
            progress_callback: Callback function(progress, message)
            
        Returns:
            Dict with output paths and statistics
        """
        start_time = time.time()
        
        # Determine output name
        if output_name is None:
            output_name = Path(input_source).stem if not self.preprocessor.is_url(input_source) else "output"
        
        output_dir = Path(self.config.output_dir) / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing: {input_source}")
        logger.info(f"Output directory: {output_dir}")
        
        # Preprocess video
        if progress_callback:
            progress_callback(0.0, "Loading video...")
        
        frame_generator, self.video_info = self.preprocessor.process(
            input_source,
            progress_callback=lambda p: progress_callback(p * 0.1, "Downloading...") if progress_callback else None
        )
        
        # Setup video writer
        width = self.video_info['width']
        height = self.video_info['height']
        fps = self.video_info['fps']
        
        # Apply max dimension
        if self.config.max_dimension:
            scale = min(1.0, self.config.max_dimension / max(width, height))
            width = int(width * scale)
            height = int(height * scale)
        
        output_video_path = str(output_dir / f"{output_name}_annotated.mp4")
        
        # Process frames
        total_frames = self.video_info.get('total_frames', 0)
        frame_count = 0
        sample_frames = []
        
        with VideoWriter(output_video_path, fps, width, height) as writer:
            for frame_id, frame in frame_generator:
                # Process frame
                annotated, tracks, stats = self.process_frame(frame, frame_id)
                
                # Write annotated frame
                writer.write(annotated)
                
                # Save sample frames for report
                if frame_count % max(1, total_frames // 5) == 0 and len(sample_frames) < 5:
                    sample_frames.append(annotated.copy())
                
                frame_count += 1
                
                # Progress update
                if progress_callback and total_frames > 0:
                    progress = 0.1 + 0.8 * (frame_count / total_frames)
                    progress_callback(progress, f"Processing frame {frame_count}/{total_frames}")
        
        # Post-processing
        if progress_callback:
            progress_callback(0.9, "Generating analytics...")
        
        # Get team assignments
        team_assignments = self.analytics.get_team_assignments()
        if team_assignments:
            self.annotator.set_team_assignments(team_assignments)
        
        # Save analytics outputs
        background_frame = sample_frames[0] if sample_frames else None
        self.analytics.save_all_outputs(str(output_dir), background_frame)
        
        # Save sample frames
        sample_frame_paths = []
        for i, frame in enumerate(sample_frames):
            frame_path = str(output_dir / f"sample_frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            sample_frame_paths.append(frame_path)
        
        # Save configuration
        config_path = str(output_dir / "config_used.json")
        config_dict = self.config.__dict__.copy()
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Compile results
        total_time = time.time() - start_time
        
        # Get speed stats
        speed_stats = self.analytics.get_speed_stats()
        
        # Get birds-eye view data (positions from last frames)
        birds_eye_path = None
        if self.config.enable_birds_eye and self.analytics.birds_eye:
            try:
                # Create a birds-eye view image
                birds_eye_img = self.analytics.birds_eye.create_pitch_image()
                birds_eye_path = str(output_dir / "birds_eye_view.png")
                cv2.imwrite(birds_eye_path, birds_eye_img)
            except Exception as e:
                logger.warning(f"Could not generate birds-eye view: {e}")
        
        results = {
            'output_dir': str(output_dir),
            'annotated_video': output_video_path,
            'heatmap': str(output_dir / "heatmap.png"),
            'count_plot': str(output_dir / "count_plot.png"),
            'metrics': str(output_dir / "metrics.json"),
            'speed_stats': str(output_dir / "speed_stats.json"),
            'birds_eye_view': birds_eye_path,
            'sample_frames': sample_frame_paths,
            'source_url': input_source,  # Store original source
            'team_assignments': team_assignments,
            'speed_data': speed_stats,
            'statistics': {
                'total_frames': frame_count,
                'total_time_seconds': total_time,
                'avg_fps': frame_count / total_time if total_time > 0 else 0,
                'video_info': self.video_info
            }
        }
        
        # Get final metrics
        if self.config.enable_metrics:
            results['tracking_metrics'] = self.analytics.get_metrics()
        
        # Generate technical report
        try:
            report_gen = ReportGenerator(str(output_dir))
            report_paths = report_gen.generate(
                results,
                config_dict,
                sample_frame_paths,
                formats=['markdown']
            )
            results['report'] = report_paths.get('markdown')
            logger.success(f"Technical report generated: {results['report']}")
        except Exception as e:
            logger.warning(f"Could not generate report: {e}")
            results['report'] = None
        
        # Save source URL metadata
        metadata = {
            'source_url': input_source,
            'source_type': 'url' if self.preprocessor.is_url(input_source) else 'file',
            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': config_dict
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        logger.success(f"Pipeline complete! Output: {output_dir}")
        logger.info(f"Processed {frame_count} frames in {total_time:.1f}s ({frame_count/total_time:.1f} FPS)")
        
        return results
    
    def cleanup(self):
        """Clean up temporary files."""
        if self._preprocessor:
            self._preprocessor.cleanup()


def run_pipeline(
    input_source: str,
    config_path: str = "config.yaml",
    output_name: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Convenience function to run the pipeline.
    
    Args:
        input_source: Video URL or file path
        config_path: Path to configuration file
        output_name: Output name
        progress_callback: Progress callback
        
    Returns:
        Pipeline results
    """
    pipeline = TrackingPipeline(config_path=config_path)
    
    try:
        results = pipeline.run(
            input_source,
            output_name=output_name,
            progress_callback=progress_callback
        )
        return results
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline <video_path_or_url>")
        sys.exit(1)
    
    input_source = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    
    results = run_pipeline(input_source, config_path)
    
    print("\n=== Pipeline Results ===")
    print(f"Output directory: {results['output_dir']}")
    print(f"Annotated video: {results['annotated_video']}")
    print(f"Processing stats: {results['statistics']}")
