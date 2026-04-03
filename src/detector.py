"""
Detector Module - YOLOv8/v9 Person Detection
============================================

This module handles person detection using Ultralytics YOLOv8/v9 models.
Optimized for sports video analysis with configurable confidence thresholds.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import torch
from ultralytics import YOLO
from loguru import logger


@dataclass
class Detection:
    """Single detection result."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


@dataclass 
class FrameDetections:
    """All detections for a single frame."""
    frame_id: int
    detections: List[Detection]
    raw_boxes: np.ndarray  # Shape: (N, 4)
    confidences: np.ndarray  # Shape: (N,)
    
    @property
    def num_detections(self) -> int:
        return len(self.detections)
    
    def to_xyxy(self) -> np.ndarray:
        """Return boxes in xyxy format."""
        return self.raw_boxes
    
    def to_xywh(self) -> np.ndarray:
        """Convert boxes to xywh format."""
        boxes = self.raw_boxes.copy()
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
        return boxes


class Detector:
    """
    YOLOv8/v9 Person Detector
    
    Features:
    - Supports YOLOv8 (n/s/m/l/x) and YOLOv9 (c/e) models
    - GPU acceleration with FP16 inference
    - Configurable confidence and NMS thresholds
    - Class filtering (person only by default)
    
    Example:
        detector = Detector(model_path="yolov8x.pt", confidence=0.35)
        detections = detector.detect(frame)
    """
    
    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        confidence_threshold: float = 0.35,
        nms_iou_threshold: float = 0.45,
        classes: List[int] = None,
        device: str = "cuda",
        imgsz: int = 1280,
        half: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            nms_iou_threshold: IoU threshold for NMS
            classes: List of class IDs to detect (0=person)
            device: Inference device (cuda/cpu/mps)
            imgsz: Input image size for inference
            half: Use FP16 inference for speed
            verbose: Show detailed logs
        """
        if classes is None:
            classes = [0]  # Default to person class
            
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.classes = classes
        self.device = self._setup_device(device)
        self.imgsz = imgsz
        self.half = half and self.device != "cpu"
        self.verbose = verbose
        
        # Load model
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Warm up model
        self._warmup()
        
        logger.success(f"Detector initialized on {self.device}")
        
    def _setup_device(self, device: str) -> str:
        """Setup and validate compute device."""
        if device == "cuda" and torch.cuda.is_available():
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            return "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            logger.info("Using Apple MPS")
            return "mps"
        else:
            if device != "cpu":
                logger.warning("GPU not available, falling back to CPU")
            return "cpu"
    
    def _warmup(self):
        """Warm up model with dummy inference."""
        logger.debug("Warming up detector...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model.predict(
            dummy,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            classes=self.classes,
            verbose=False
        )
        
    def detect(
        self, 
        frame: np.ndarray,
        frame_id: int = 0
    ) -> FrameDetections:
        """
        Detect persons in a single frame.
        
        Args:
            frame: BGR image as numpy array (H, W, 3)
            frame_id: Optional frame identifier
            
        Returns:
            FrameDetections object with all detections
        """
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            classes=self.classes,
            imgsz=self.imgsz,
            half=self.half,
            verbose=self.verbose,
            device=self.device
        )
        
        # Parse results
        detections = []
        boxes_list = []
        confidences_list = []
        
        if len(results) > 0 and results[0].boxes is not None:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
                detection = Detection(
                    bbox=box,
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=self.model.names[cls_id]
                )
                detections.append(detection)
                boxes_list.append(box)
                confidences_list.append(conf)
        
        # Convert to arrays
        raw_boxes = np.array(boxes_list) if boxes_list else np.empty((0, 4))
        confidences = np.array(confidences_list) if confidences_list else np.empty(0)
        
        return FrameDetections(
            frame_id=frame_id,
            detections=detections,
            raw_boxes=raw_boxes,
            confidences=confidences
        )
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        start_frame_id: int = 0
    ) -> List[FrameDetections]:
        """
        Detect persons in a batch of frames.
        
        Args:
            frames: List of BGR images
            start_frame_id: Starting frame ID
            
        Returns:
            List of FrameDetections for each frame
        """
        results = self.model.predict(
            frames,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            classes=self.classes,
            imgsz=self.imgsz,
            half=self.half,
            verbose=self.verbose,
            device=self.device,
            stream=True
        )
        
        frame_detections = []
        for i, result in enumerate(results):
            detections = []
            boxes_list = []
            confidences_list = []
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confs, cls_ids):
                    detection = Detection(
                        bbox=box,
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=self.model.names[cls_id]
                    )
                    detections.append(detection)
                    boxes_list.append(box)
                    confidences_list.append(conf)
            
            raw_boxes = np.array(boxes_list) if boxes_list else np.empty((0, 4))
            confidences = np.array(confidences_list) if confidences_list else np.empty(0)
            
            frame_detections.append(FrameDetections(
                frame_id=start_frame_id + i,
                detections=detections,
                raw_boxes=raw_boxes,
                confidences=confidences
            ))
        
        return frame_detections
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for reporting."""
        return {
            "model_type": "YOLO",
            "confidence_threshold": self.confidence_threshold,
            "nms_iou_threshold": self.nms_iou_threshold,
            "classes": self.classes,
            "device": self.device,
            "imgsz": self.imgsz,
            "half_precision": self.half
        }


class ModelComparator:
    """
    Compare detection performance between multiple YOLO models.
    
    Useful for evaluating YOLOv8 vs YOLOv9 performance.
    """
    
    def __init__(
        self,
        model_paths: List[str],
        **detector_kwargs
    ):
        """
        Initialize comparator with multiple models.
        
        Args:
            model_paths: List of paths to YOLO model weights
            **detector_kwargs: Arguments passed to Detector
        """
        self.detectors = {}
        for path in model_paths:
            name = Path(path).stem
            self.detectors[name] = Detector(model_path=path, **detector_kwargs)
            
    def compare_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0
    ) -> Dict[str, FrameDetections]:
        """
        Run all detectors on a single frame.
        
        Returns:
            Dict mapping model name to detections
        """
        results = {}
        for name, detector in self.detectors.items():
            results[name] = detector.detect(frame, frame_id)
        return results
    
    def compare_stats(
        self,
        results: Dict[str, List[FrameDetections]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate comparison statistics.
        
        Args:
            results: Dict mapping model name to list of frame detections
            
        Returns:
            Statistics for each model
        """
        stats = {}
        for name, frame_dets in results.items():
            all_confs = []
            total_dets = 0
            
            for fd in frame_dets:
                total_dets += fd.num_detections
                all_confs.extend(fd.confidences.tolist())
            
            stats[name] = {
                "total_detections": total_dets,
                "avg_detections_per_frame": total_dets / len(frame_dets) if frame_dets else 0,
                "avg_confidence": float(np.mean(all_confs)) if all_confs else 0,
                "min_confidence": float(np.min(all_confs)) if all_confs else 0,
                "max_confidence": float(np.max(all_confs)) if all_confs else 0,
                "std_confidence": float(np.std(all_confs)) if all_confs else 0
            }
        
        return stats
