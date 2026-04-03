"""
Preprocessor Module - Video Download & Frame Extraction
=======================================================

This module handles video input from URLs or files, downloading via yt-dlp
and extracting frames using ffmpeg or OpenCV.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator, Optional, Tuple, Dict, Any, List
import cv2
import numpy as np
from loguru import logger
import subprocess
import json

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    logger.warning("yt-dlp not available, URL downloads disabled")

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    logger.warning("ffmpeg-python not available, using OpenCV for frame extraction")


class VideoDownloader:
    """Download videos from URLs using yt-dlp."""
    
    def __init__(self, output_dir: str = "temp"):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory to save downloaded videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download(
        self,
        url: str,
        max_resolution: int = 1080,
        progress_callback: Optional[callable] = None
    ) -> Optional[str]:
        """
        Download video from URL.
        
        Args:
            url: Video URL (YouTube, etc.)
            max_resolution: Maximum resolution to download
            progress_callback: Optional progress callback
            
        Returns:
            Path to downloaded video or None on failure
        """
        if not YTDLP_AVAILABLE:
            logger.error("yt-dlp not installed")
            return None
        
        output_template = str(self.output_dir / "%(title)s.%(ext)s")
        
        ydl_opts = {
            'format': f'bestvideo[height<={max_resolution}]+bestaudio/best[height<={max_resolution}]',
            'outtmpl': output_template,
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
        }
        
        if progress_callback:
            def progress_hook(d):
                if d['status'] == 'downloading':
                    total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                    downloaded = d.get('downloaded_bytes', 0)
                    if total > 0:
                        progress_callback(downloaded / total)
                elif d['status'] == 'finished':
                    progress_callback(1.0)
            
            ydl_opts['progress_hooks'] = [progress_hook]
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                
                # Handle extension changes from merging
                if not os.path.exists(filename):
                    base = os.path.splitext(filename)[0]
                    for ext in ['.mp4', '.mkv', '.webm']:
                        if os.path.exists(base + ext):
                            filename = base + ext
                            break
                
                logger.success(f"Downloaded: {filename}")
                return filename
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None
    
    def get_video_info(self, url: str) -> Optional[Dict[str, Any]]:
        """Get video information without downloading."""
        if not YTDLP_AVAILABLE:
            return None
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title'),
                    'duration': info.get('duration'),
                    'resolution': f"{info.get('width', '?')}x{info.get('height', '?')}",
                    'fps': info.get('fps'),
                    'filesize': info.get('filesize_approx')
                }
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return None


class FrameExtractor:
    """Extract frames from video files."""
    
    def __init__(
        self,
        frame_skip: int = 1,
        max_dimension: Optional[int] = None,
        target_fps: Optional[float] = None
    ):
        """
        Initialize frame extractor.
        
        Args:
            frame_skip: Process every Nth frame
            max_dimension: Resize if larger than this
            target_fps: Target FPS for extraction
        """
        self.frame_skip = max(1, frame_skip)
        self.max_dimension = max_dimension
        self.target_fps = target_fps
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata."""
        cap = cv2.VideoCapture(video_path)
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if needed."""
        if self.max_dimension is None:
            return frame
        
        h, w = frame.shape[:2]
        if max(h, w) <= self.max_dimension:
            return frame
        
        scale = self.max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(frame, (new_w, new_h))
    
    def extract_frames_opencv(
        self,
        video_path: str,
        progress_callback: Optional[callable] = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Extract frames using OpenCV.
        
        Args:
            video_path: Path to video file
            progress_callback: Optional progress callback
            
        Yields:
            (frame_id, frame) tuples
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame skip based on target FPS
        effective_skip = self.frame_skip
        if self.target_fps and fps > self.target_fps:
            effective_skip = max(effective_skip, int(fps / self.target_fps))
        
        frame_id = 0
        extracted = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_id % effective_skip == 0:
                frame = self._resize_frame(frame)
                yield frame_id, frame
                extracted += 1
            
            frame_id += 1
            
            if progress_callback and total_frames > 0:
                progress_callback(frame_id / total_frames)
        
        cap.release()
        logger.info(f"Extracted {extracted} frames from {total_frames} total")
    
    def extract_frames_ffmpeg(
        self,
        video_path: str,
        output_dir: str,
        progress_callback: Optional[callable] = None
    ) -> List[str]:
        """
        Extract frames using ffmpeg (faster for large videos).
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            progress_callback: Optional progress callback
            
        Returns:
            List of frame file paths
        """
        if not FFMPEG_AVAILABLE:
            logger.warning("ffmpeg-python not available, use extract_frames_opencv instead")
            return []
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get video info
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        total_frames = int(video_info.get('nb_frames', 0))
        
        # Build ffmpeg command
        output_pattern = str(output_path / "frame_%06d.jpg")
        
        stream = ffmpeg.input(video_path)
        
        if self.frame_skip > 1:
            stream = stream.filter('select', f'not(mod(n,{self.frame_skip}))')
        
        if self.max_dimension:
            stream = stream.filter(
                'scale',
                f'min({self.max_dimension},iw)',
                f'min({self.max_dimension},ih)',
                force_original_aspect_ratio='decrease'
            )
        
        try:
            stream.output(output_pattern, vsync='vfr').run(quiet=True)
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e}")
            return []
        
        # Get list of extracted frames
        frame_files = sorted(output_path.glob("frame_*.jpg"))
        logger.info(f"Extracted {len(frame_files)} frames to {output_dir}")
        
        return [str(f) for f in frame_files]
    
    def frames_from_directory(
        self,
        frame_dir: str
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Load frames from a directory of images.
        
        Args:
            frame_dir: Directory containing frame images
            
        Yields:
            (frame_id, frame) tuples
        """
        frame_path = Path(frame_dir)
        frame_files = sorted(frame_path.glob("*.jpg")) + sorted(frame_path.glob("*.png"))
        
        for i, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                yield i, frame


class VideoPreprocessor:
    """
    Main video preprocessing pipeline.
    
    Handles:
    - URL detection and downloading
    - Frame extraction with configurable FPS
    - Automatic resizing
    """
    
    def __init__(
        self,
        frame_skip: int = 1,
        max_dimension: Optional[int] = 1920,
        target_fps: Optional[float] = None,
        temp_dir: str = "temp"
    ):
        """
        Initialize preprocessor.
        
        Args:
            frame_skip: Process every Nth frame
            max_dimension: Maximum frame dimension
            target_fps: Target FPS for processing
            temp_dir: Temporary directory for downloads
        """
        self.downloader = VideoDownloader(output_dir=temp_dir)
        self.extractor = FrameExtractor(
            frame_skip=frame_skip,
            max_dimension=max_dimension,
            target_fps=target_fps
        )
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def is_url(self, input_path: str) -> bool:
        """Check if input is a URL."""
        return input_path.startswith(('http://', 'https://', 'www.'))
    
    def process(
        self,
        input_source: str,
        progress_callback: Optional[callable] = None
    ) -> Tuple[Generator[Tuple[int, np.ndarray], None, None], Dict[str, Any]]:
        """
        Process video from URL or file path.
        
        Args:
            input_source: Video URL or file path
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (frame generator, video info dict)
        """
        # Download if URL
        if self.is_url(input_source):
            logger.info(f"Downloading video from URL: {input_source}")
            video_path = self.downloader.download(
                input_source,
                progress_callback=progress_callback
            )
            if video_path is None:
                raise RuntimeError("Failed to download video")
        else:
            video_path = input_source
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Get video info
        video_info = self.extractor.get_video_info(video_path)
        video_info['source'] = input_source
        video_info['local_path'] = video_path
        
        logger.info(f"Video info: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.1f}fps, {video_info['duration']:.1f}s")
        
        # Create frame generator
        frame_generator = self.extractor.extract_frames_opencv(
            video_path,
            progress_callback=progress_callback
        )
        
        return frame_generator, video_info
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if self.temp_dir.exists():
            for item in self.temp_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            logger.info("Cleaned up temporary files")


def detect_court_lines(
    frame: np.ndarray,
    min_line_length: int = 100,
    max_line_gap: int = 10
) -> List[Tuple[int, int, int, int]]:
    """
    Detect court/field lines using Hough transform.
    
    Useful for automatic homography point detection.
    
    Args:
        frame: Input frame
        min_line_length: Minimum line length in pixels
        max_line_gap: Maximum gap between line segments
        
    Returns:
        List of (x1, y1, x2, y2) line coordinates
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        return []
    
    return [tuple(line[0]) for line in lines]


def find_corner_candidates(
    lines: List[Tuple[int, int, int, int]],
    frame_shape: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Find potential court corner points from detected lines.
    
    Args:
        lines: List of detected lines
        frame_shape: (height, width) of frame
        
    Returns:
        List of (x, y) corner candidates
    """
    if len(lines) < 2:
        return []
    
    corners = []
    
    # Find line intersections
    for i, (x1, y1, x2, y2) in enumerate(lines):
        for j, (x3, y3, x4, y4) in enumerate(lines[i+1:], i+1):
            # Line 1: (x1,y1) to (x2,y2)
            # Line 2: (x3,y3) to (x4,y4)
            
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-6:
                continue  # Parallel lines
            
            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
            
            px = x1 + t*(x2-x1)
            py = y1 + t*(y2-y1)
            
            # Check if intersection is within frame
            h, w = frame_shape
            if 0 <= px <= w and 0 <= py <= h:
                corners.append((int(px), int(py)))
    
    return corners
