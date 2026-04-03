"""
Re-Identification Module - ResNet Appearance Embedding
======================================================

This module extracts appearance embeddings using torchvision pretrained
models (ResNet/EfficientNet) for persistent track identification across
occlusions and camera cuts.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import cv2
import torch
import torch.nn.functional as F
from loguru import logger

try:
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
    logger.success("torchvision ReID backend loaded")
except ImportError:
    TORCHVISION_AVAILABLE = False
    logger.warning("torchvision not available, ReID features disabled")


@dataclass
class Embedding:
    """Single appearance embedding."""
    track_id: int
    frame_id: int
    embedding: np.ndarray
    confidence: float


@dataclass
class Gallery:
    """Gallery of embeddings for a single track."""
    track_id: int
    embeddings: List[np.ndarray] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)
    max_size: int = 100
    
    def add(self, embedding: np.ndarray, frame_id: int):
        """Add embedding to gallery."""
        self.embeddings.append(embedding)
        self.frame_ids.append(frame_id)
        
        # Keep only most recent embeddings
        if len(self.embeddings) > self.max_size:
            self.embeddings = self.embeddings[-self.max_size:]
            self.frame_ids = self.frame_ids[-self.max_size:]
    
    def get_mean_embedding(self) -> np.ndarray:
        """Get mean embedding for matching."""
        if not self.embeddings:
            return np.zeros(576)  # MobileNetV3 small dim
        return np.mean(self.embeddings, axis=0)
    
    def get_recent_embedding(self) -> np.ndarray:
        """Get most recent embedding."""
        if not self.embeddings:
            return np.zeros(576)  # MobileNetV3 small dim
        return self.embeddings[-1]


class ResNetExtractor:
    """MobileNetV3 feature extractor using torchvision - optimized for speed."""
    
    def __init__(
        self,
        model_name: str = "mobilenet_v3_small",
        device: str = "cuda",
        input_size: Tuple[int, int] = (128, 64)  # Smaller input for speed
    ):
        """
        Initialize feature extractor.
        
        Args:
            model_name: Model variant (mobilenet_v3_small, resnet18, efficientnet_b0)
            device: Compute device
            input_size: Input size (height, width) - smaller = faster
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.embedding_dim = 512
        
        if TORCHVISION_AVAILABLE:
            try:
                # Use lightweight models for speed
                if model_name in ["mobilenet_v3_small", "osnet_x0_25", "osnet_x0_5"]:
                    base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
                    self.embedding_dim = 576
                    # Get features before classifier
                    self.model = torch.nn.Sequential(
                        base_model.features,
                        base_model.avgpool
                    )
                elif model_name in ["mobilenet_v3_large"]:
                    base_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
                    self.embedding_dim = 960
                    self.model = torch.nn.Sequential(
                        base_model.features,
                        base_model.avgpool
                    )
                elif model_name == "resnet18":
                    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                    self.embedding_dim = 512
                    self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
                elif model_name == "efficientnet_b0":
                    base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                    self.embedding_dim = 1280
                    self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
                else:
                    # Default to MobileNetV3 for speed
                    base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
                    self.embedding_dim = 576
                    self.model = torch.nn.Sequential(
                        base_model.features,
                        base_model.avgpool
                    )
                
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Image transforms
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((input_size[0], input_size[1])),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                
                logger.success(f"Feature extractor ({model_name}) loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
        else:
            self.model = None
            logger.warning("Using dummy features (torchvision not available)")
    
    def extract(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from image crops.
        
        Args:
            crops: List of BGR images
            
        Returns:
            (N, embedding_dim) array of normalized embeddings
        """
        if not crops:
            return np.empty((0, self.embedding_dim))
        
        if self.model is None:
            # Return dummy embeddings
            return np.random.randn(len(crops), self.embedding_dim).astype(np.float32)
        
        # Preprocess crops
        batch = []
        for crop in crops:
            if crop.size == 0:
                crop = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb)
            batch.append(tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(batch).to(self.device)
        
        # Extract features
        try:
            with torch.no_grad():
                features = self.model(batch_tensor)
                features = features.squeeze(-1).squeeze(-1)  # Remove spatial dims
                features = features.cpu().numpy()
            
            # L2 normalize
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / np.maximum(norms, 1e-6)
            
            return features
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.random.randn(len(crops), self.embedding_dim).astype(np.float32)
    
    def extract_single(self, crop: np.ndarray) -> np.ndarray:
        """Extract features from a single crop."""
        return self.extract([crop])[0]


class ReIDModule:
    """
    Re-Identification module for persistent track IDs.
    
    Uses lightweight MobileNetV3 embeddings and cosine similarity to re-identify
    subjects across occlusions, camera cuts, and re-entries.
    """
    
    def __init__(
        self,
        model_name: str = "mobilenet_v3_small",
        embedding_dim: int = 576,
        gallery_size: int = 50,  # Reduced for speed
        similarity_threshold: float = 0.60,
        update_interval: int = 10,  # Update less frequently
        device: str = "cuda"
    ):
        """
        Initialize ReID module.
        
        Args:
            model_name: Model variant (mobilenet_v3_small for speed, resnet18 for accuracy)
            embedding_dim: Embedding dimension (auto-set based on model)
            gallery_size: Max embeddings per track
            similarity_threshold: Threshold for ID matching
            update_interval: Update gallery every N frames (higher = faster)
            device: Compute device
        """
        self.gallery_size = gallery_size
        self.similarity_threshold = similarity_threshold
        self.update_interval = update_interval
        
        # Initialize extractor with lightweight model
        self.extractor = ResNetExtractor(
            model_name=model_name,
            device=device,
            input_size=(128, 64)  # Smaller input for speed
        )
        self.embedding_dim = self.extractor.embedding_dim
        
        # Track galleries
        self.galleries: Dict[int, Gallery] = {}
        
        # Lost tracks (for re-identification)
        self.lost_galleries: Dict[int, Gallery] = {}
        
        # ID mapping
        self.id_remapping: Dict[int, int] = {}
        
        # Frame counter per track
        self.track_frame_count: Dict[int, int] = defaultdict(int)
        
        logger.success("ReID module initialized")
    
    def extract_crops(
        self,
        frame: np.ndarray,
        boxes: np.ndarray
    ) -> List[np.ndarray]:
        """
        Extract image crops from frame.
        
        Args:
            frame: Full frame image
            boxes: (N, 4) array of [x1, y1, x2, y2]
            
        Returns:
            List of cropped images
        """
        crops = []
        h, w = frame.shape[:2]
        
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            
            # Clip to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((128, 64, 3), dtype=np.uint8))
                continue
            
            crop = frame[y1:y2, x1:x2].copy()
            crops.append(crop)
        
        return crops
    
    def compute_similarity(
        self,
        embedding: np.ndarray,
        gallery: Gallery
    ) -> float:
        """Compute cosine similarity between embedding and gallery."""
        gallery_embedding = gallery.get_mean_embedding()
        
        # Cosine similarity
        sim = np.dot(embedding, gallery_embedding)
        sim = sim / (np.linalg.norm(embedding) * np.linalg.norm(gallery_embedding) + 1e-6)
        
        return float(sim)
    
    def find_matching_lost_track(
        self,
        embedding: np.ndarray
    ) -> Optional[int]:
        """
        Find matching lost track for re-identification.
        
        Args:
            embedding: Query embedding
            
        Returns:
            Matching track ID or None
        """
        best_match = None
        best_sim = self.similarity_threshold
        
        for track_id, gallery in self.lost_galleries.items():
            sim = self.compute_similarity(embedding, gallery)
            if sim > best_sim:
                best_sim = sim
                best_match = track_id
        
        return best_match
    
    def update(
        self,
        frame: np.ndarray,
        track_ids: List[int],
        boxes: np.ndarray,
        frame_id: int
    ) -> Dict[int, int]:
        """
        Update ReID galleries and perform re-identification.
        
        Args:
            frame: Current frame
            track_ids: List of track IDs from tracker
            boxes: (N, 4) array of bounding boxes
            frame_id: Current frame number
            
        Returns:
            ID remapping dict (original_id -> unified_id)
        """
        if len(track_ids) == 0:
            return {}
        
        # Extract crops
        crops = self.extract_crops(frame, boxes)
        
        # Extract embeddings
        embeddings = self.extractor.extract(crops)
        
        # Process each track
        remapping = {}
        current_track_ids = set()
        
        for i, (track_id, embedding) in enumerate(zip(track_ids, embeddings)):
            current_track_ids.add(track_id)
            self.track_frame_count[track_id] += 1
            
            # Check if this is a new track that might match a lost one
            if track_id not in self.galleries:
                # Try to find matching lost track
                matched_id = self.find_matching_lost_track(embedding)
                
                if matched_id is not None:
                    # Re-identify: merge with lost track
                    self.galleries[track_id] = self.lost_galleries.pop(matched_id)
                    self.id_remapping[track_id] = matched_id
                    remapping[track_id] = matched_id
                    logger.debug(f"ReID: Track {track_id} -> {matched_id}")
                else:
                    # New track
                    self.galleries[track_id] = Gallery(
                        track_id=track_id,
                        max_size=self.gallery_size
                    )
                    remapping[track_id] = track_id
            else:
                remapping[track_id] = self.id_remapping.get(track_id, track_id)
            
            # Update gallery periodically
            if self.track_frame_count[track_id] % self.update_interval == 0:
                self.galleries[track_id].add(embedding, frame_id)
        
        # Move missing tracks to lost galleries
        active_ids = set(self.galleries.keys())
        for track_id in active_ids - current_track_ids:
            if track_id in self.galleries:
                self.lost_galleries[track_id] = self.galleries.pop(track_id)
        
        # Clean up old lost galleries (keep only recent)
        max_lost = 50
        if len(self.lost_galleries) > max_lost:
            # Remove oldest
            sorted_lost = sorted(
                self.lost_galleries.items(),
                key=lambda x: x[1].frame_ids[-1] if x[1].frame_ids else 0
            )
            for track_id, _ in sorted_lost[:-max_lost]:
                del self.lost_galleries[track_id]
        
        return remapping
    
    def get_unified_id(self, track_id: int) -> int:
        """Get unified ID for a track (handles re-identifications)."""
        return self.id_remapping.get(track_id, track_id)
    
    def get_gallery_stats(self) -> Dict[str, Any]:
        """Get gallery statistics for reporting."""
        return {
            "active_galleries": len(self.galleries),
            "lost_galleries": len(self.lost_galleries),
            "total_remappings": len(self.id_remapping),
            "similarity_threshold": self.similarity_threshold,
            "gallery_size": self.gallery_size
        }
    
    def get_reid_info(self) -> Dict[str, Any]:
        """Get ReID configuration for reporting."""
        return {
            "model": "ResNet/EfficientNet",
            "embedding_dim": self.embedding_dim,
            "gallery_size": self.gallery_size,
            "similarity_threshold": self.similarity_threshold,
            "update_interval": self.update_interval,
            "active_tracks": len(self.galleries),
            "lost_tracks": len(self.lost_galleries),
            "reidentifications": len(self.id_remapping)
        }


class ColorHistogramReID:
    """
    Simple color histogram-based ReID as fallback.
    
    Uses HSV color histograms for appearance matching.
    Useful when torchreid is not available.
    """
    
    def __init__(
        self,
        bins: Tuple[int, int, int] = (8, 8, 8),
        similarity_threshold: float = 0.7
    ):
        self.bins = bins
        self.similarity_threshold = similarity_threshold
        self.galleries: Dict[int, List[np.ndarray]] = {}
    
    def extract_histogram(self, crop: np.ndarray) -> np.ndarray:
        """Extract HSV histogram from crop."""
        if crop.size == 0:
            return np.zeros(np.prod(self.bins))
        
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            self.bins, [0, 180, 0, 256, 0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def compare_histograms(
        self,
        hist1: np.ndarray,
        hist2: np.ndarray
    ) -> float:
        """Compare two histograms using correlation."""
        return cv2.compareHist(
            hist1.astype(np.float32),
            hist2.astype(np.float32),
            cv2.HISTCMP_CORREL
        )
    
    def update(
        self,
        frame: np.ndarray,
        track_ids: List[int],
        boxes: np.ndarray
    ) -> Dict[int, int]:
        """Update galleries with new observations."""
        remapping = {}
        
        for track_id, box in zip(track_ids, boxes):
            x1, y1, x2, y2 = box.astype(int)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            hist = self.extract_histogram(crop)
            
            if track_id not in self.galleries:
                self.galleries[track_id] = []
            
            self.galleries[track_id].append(hist)
            
            # Keep only recent histograms
            if len(self.galleries[track_id]) > 30:
                self.galleries[track_id] = self.galleries[track_id][-30:]
            
            remapping[track_id] = track_id
        
        return remapping
