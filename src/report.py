"""
Report Generator Module - Automated Technical Report
====================================================

This module generates technical reports in Markdown and PDF formats,
documenting the pipeline configuration, results, and analysis.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from loguru import logger

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False


class ReportGenerator:
    """
    Generate technical reports for pipeline runs.
    
    Creates Markdown and optional PDF reports with:
    - Pipeline configuration
    - Model and tracker details
    - ReID approach
    - Results and metrics
    - Sample frames
    - Challenges and improvements
    """
    
    def __init__(
        self,
        output_dir: str,
        api_key: Optional[str] = None
    ):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
            api_key: Optional API key for AI-assisted report generation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
    
    def generate_markdown(
        self,
        results: Dict[str, Any],
        config: Dict[str, Any],
        sample_frames: List[str],
        title: str = "Multi-Object Tracking Pipeline Report"
    ) -> str:
        """
        Generate Markdown report.
        
        Args:
            results: Pipeline results dict
            config: Pipeline configuration dict
            sample_frames: Paths to sample frame images
            title: Report title
            
        Returns:
            Markdown content as string
        """
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract statistics
        stats = results.get('statistics', {})
        video_info = stats.get('video_info', {})
        metrics = results.get('tracking_metrics', {})
        source_url = results.get('source_url', 'N/A')
        
        # Build report
        report = f"""# {title}

**Generated:** {timestamp}

---

## 1. Executive Summary

This report documents the results of running a multi-object detection and tracking pipeline on sports/event video footage. The pipeline detects persons frame-by-frame using deep learning models, assigns persistent unique IDs using Kalman filter tracking, and re-identifies subjects across occlusions using appearance embeddings.

### Key Results
- **Total Frames Processed:** {stats.get('total_frames', 'N/A')}
- **Processing Time:** {stats.get('total_time_seconds', 0):.1f} seconds
- **Average FPS:** {stats.get('avg_fps', 0):.1f}
- **Total Tracks:** {metrics.get('total_tracks', 'N/A')}
- **Average Track Length:** {metrics.get('avg_track_length', 0):.1f} frames

---

## 2. Input Video Information

| Property | Value |
|----------|-------|
| **Source URL/Path** | {source_url} |
| **Resolution** | {video_info.get('width', '?')}x{video_info.get('height', '?')} |
| **Frame Rate** | {video_info.get('fps', 0):.1f} FPS |
| **Duration** | {video_info.get('duration', 0):.1f} seconds |
| **Total Frames** | {video_info.get('total_frames', 'N/A')} |

---

## 3. Model & Configuration

### 3.1 Detection Model

The detection stage uses **YOLO (You Only Look Once)** architecture for real-time person detection.

| Parameter | Value |
|-----------|-------|
| **Model** | {config.get('detection_model', 'yolov8x.pt')} |
| **Confidence Threshold** | {config.get('confidence_threshold', 0.35)} |
| **NMS IoU Threshold** | {config.get('nms_iou_threshold', 0.45)} |
| **Input Size** | {config.get('imgsz', 1280)} |
| **Device** | {config.get('device', 'cuda')} |

**Why YOLO?**
- State-of-the-art real-time object detection
- Excellent balance of speed and accuracy
- Native support for person detection (COCO class 0)
- Optimized for GPU inference with FP16 support

### 3.2 Tracking Algorithm

The pipeline uses a **Kalman Filter** with Hungarian algorithm for robust tracking:

| Component | Purpose |
|-----------|---------|
| **Kalman Filter** | Predicts object positions between frames |
| **Hungarian Algorithm** | Optimal detection-to-track assignment |
| **IoU Matching** | Associates detections with existing tracks |

| Parameter | Value |
|-----------|-------|
| **Primary Tracker** | {config.get('primary_tracker', 'kalman')} |
| **Frame Rate** | Adaptive |

**Why Kalman Filter?**
- Smooth trajectory prediction during occlusions
- Computationally efficient (pure NumPy)
- No external dependencies required
- Proven effective for multi-object tracking

### 3.3 Re-Identification (ReID)

The ReID module ensures **persistent ID assignment** across occlusions and re-entries.

| Parameter | Value |
|-----------|-------|
| **Model** | torchvision ({config.get('reid_model', 'mobilenet_v3_small')}) |
| **Embedding Dimension** | 576 (MobileNetV3) / 2048 (ResNet) |
| **Similarity Threshold** | {config.get('similarity_threshold', 0.60)} |
| **Enabled** | {config.get('reid_enabled', True)} |

**How ReID Works:**
1. Extract appearance embedding for each detected person using pretrained CNN
2. Maintain a gallery of embeddings per track ID
3. When a track is lost and re-appears, compare against gallery using cosine similarity
4. If similarity exceeds threshold, merge IDs (re-identification successful)

**Why torchvision Models?**
- Pretrained on ImageNet - strong general features
- MobileNetV3 is fast and lightweight
- No additional package dependencies
- Good generalization to sports scenarios

---

## 4. ID Consistency Strategy

Maintaining consistent IDs across the video is achieved through a multi-layered approach:

### Layer 1: Frame-to-Frame Tracking
- Kalman filter predicts next position based on velocity
- Hungarian algorithm matches detections to tracks optimally
- IoU-based association for reliable matching

### Layer 2: Appearance Embeddings
- torchvision CNN extracts visual features from each person crop
- Features are L2-normalized for cosine similarity comparison
- Gallery maintains recent embeddings per track

### Layer 3: Re-Identification
- Lost tracks are moved to a "lost gallery"
- New tracks are compared against lost gallery
- High similarity triggers ID merge (re-identification)

### Layer 4: Track Management
- Tracks are created for unmatched detections
- Lost tracks are kept for configurable number of frames
- Aged tracks are removed to prevent ID explosion

---

## 5. Results & Metrics

### 5.1 Tracking Performance

| Metric | Value |
|--------|-------|
| **Total Tracks** | {metrics.get('total_tracks', 'N/A')} |
| **Average Track Length** | {metrics.get('avg_track_length', 0):.1f} frames |
| **Min Track Length** | {metrics.get('min_track_length', 'N/A')} frames |
| **Max Track Length** | {metrics.get('max_track_length', 'N/A')} frames |
| **Total Detections** | {metrics.get('total_detections', 'N/A')} |
| **Avg Detections/Frame** | {metrics.get('avg_detections_per_frame', 0):.1f} |
| **ID Switches** | {metrics.get('id_switches', 'N/A')} |
| **Fragmentations** | {metrics.get('fragmentations', 'N/A')} |
| **Fragmentation Rate** | {metrics.get('fragmentation_rate', 0):.3f} |

### 5.2 Processing Performance

| Metric | Value |
|--------|-------|
| **Total Processing Time** | {stats.get('total_time_seconds', 0):.1f} seconds |
| **Average FPS** | {stats.get('avg_fps', 0):.1f} |
| **Frames Processed** | {stats.get('total_frames', 'N/A')} |

---

## 6. Sample Annotated Frames

"""
        
        # Add sample frame references
        for i, frame_path in enumerate(sample_frames[:5]):
            frame_name = Path(frame_path).name
            report += f"### Frame {i+1}\n"
            report += f"![Sample Frame {i+1}]({frame_name})\n\n"
        
        report += """---

## 7. Challenges Faced

### 7.1 Occlusion Handling
- **Challenge:** Players frequently occlude each other in sports footage
- **Solution:** OSNet ReID recovers IDs after occlusion ends

### 7.2 Camera Motion
- **Challenge:** Broadcast cameras pan and zoom rapidly
- **Solution:** BoT-SORT's ECC camera motion compensation

### 7.3 Similar Appearances
- **Challenge:** Players on the same team look similar
- **Solution:** Team clustering separates teams, trajectory context aids distinction

### 7.4 Crowded Scenes
- **Challenge:** Dense player clusters in set pieces
- **Solution:** ByteTrack's low-confidence detection association

### 7.5 Scale Variations
- **Challenge:** Players appear at different scales due to camera distance
- **Solution:** Multi-scale detection with YOLOv8

---

## 8. Potential Failure Cases

1. **Complete occlusion for extended periods** (>30 frames) may cause ID loss
2. **Extremely similar appearance** between two individuals
3. **Rapid camera cuts** to different field locations
4. **Very crowded scenes** with >20 overlapping persons
5. **Poor lighting conditions** affecting appearance features

---

## 9. Improvement Suggestions

### Short-term Improvements
- [ ] Fine-tune confidence threshold per video type
- [ ] Implement adaptive ReID threshold based on gallery quality
- [ ] Add motion-based features to complement appearance

### Long-term Improvements
- [ ] Train custom ReID model on sports-specific data
- [ ] Implement pose estimation for better identity features
- [ ] Add temporal consistency constraints (smoothing)
- [ ] Implement camera calibration for accurate speed estimation

---

## 10. Output Files

| File | Description |
|------|-------------|
| `*_annotated.mp4` | Annotated output video with bounding boxes and IDs |
| `heatmap.png` | Movement heatmap showing activity density |
| `count_plot.png` | Object count over time graph |
| `metrics.json` | Detailed tracking metrics |
| `speed_stats.json` | Per-track speed statistics |
| `config_used.json` | Configuration used for this run |

---

## 11. Technical Specifications

### Software Stack
- **Detection:** Ultralytics YOLOv8
- **Tracking:** Kalman Filter + Hungarian Algorithm (SciPy)
- **ReID:** torchvision (MobileNetV3/ResNet)
- **Annotation:** OpenCV
- **Video Processing:** OpenCV, yt-dlp
- **Analytics:** NumPy, SciPy, scikit-learn, Matplotlib

### Hardware Requirements
- GPU: NVIDIA GPU with CUDA support (recommended, but CPU works)
- RAM: 8GB minimum, 16GB recommended
- Storage: ~500MB per minute of processed video

---

*Report generated automatically by the Multi-Object Tracking Pipeline*
"""
        
        return report
    
    def save_markdown(
        self,
        results: Dict[str, Any],
        config: Dict[str, Any],
        sample_frames: List[str],
        filename: str = "report.md"
    ) -> str:
        """
        Save Markdown report to file.
        
        Args:
            results: Pipeline results
            config: Pipeline configuration
            sample_frames: Paths to sample frames
            filename: Output filename
            
        Returns:
            Path to saved report
        """
        content = self.generate_markdown(results, config, sample_frames)
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.success(f"Markdown report saved: {output_path}")
        return str(output_path)
    
    def save_pdf(
        self,
        results: Dict[str, Any],
        config: Dict[str, Any],
        sample_frames: List[str],
        filename: str = "report.pdf"
    ) -> Optional[str]:
        """
        Save PDF report.
        
        Args:
            results: Pipeline results
            config: Pipeline configuration
            sample_frames: Paths to sample frames
            filename: Output filename
            
        Returns:
            Path to saved report or None if PDF generation unavailable
        """
        if not MARKDOWN_AVAILABLE or not WEASYPRINT_AVAILABLE:
            logger.warning("PDF generation requires 'markdown' and 'weasyprint' packages")
            return None
        
        # Generate markdown
        md_content = self.generate_markdown(results, config, sample_frames)
        
        # Convert to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code']
        )
        
        # Add styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    color: #333;
                }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                h3 {{ color: #7f8c8d; }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                code {{
                    background-color: #f4f4f4;
                    padding: 2px 6px;
                    border-radius: 3px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    margin: 10px 0;
                }}
                hr {{ border: none; border-top: 1px solid #eee; margin: 30px 0; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Generate PDF
        output_path = self.output_dir / filename
        
        try:
            HTML(string=styled_html, base_url=str(self.output_dir)).write_pdf(str(output_path))
            logger.success(f"PDF report saved: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return None
    
    def generate(
        self,
        results: Dict[str, Any],
        config: Dict[str, Any],
        sample_frames: Optional[List[str]] = None,
        formats: List[str] = ["markdown", "pdf"]
    ) -> Dict[str, str]:
        """
        Generate report in specified formats.
        
        Args:
            results: Pipeline results
            config: Pipeline configuration
            sample_frames: Paths to sample frames
            formats: List of formats to generate ("markdown", "pdf")
            
        Returns:
            Dict mapping format to output path
        """
        if sample_frames is None:
            sample_frames = results.get('sample_frames', [])
        
        outputs = {}
        
        if "markdown" in formats:
            outputs["markdown"] = self.save_markdown(results, config, sample_frames)
        
        if "pdf" in formats:
            pdf_path = self.save_pdf(results, config, sample_frames)
            if pdf_path:
                outputs["pdf"] = pdf_path
        
        return outputs


def generate_report(
    output_dir: str,
    results: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, str]:
    """
    Convenience function to generate reports.
    
    Args:
        output_dir: Output directory
        results: Pipeline results
        config: Pipeline configuration
        
    Returns:
        Dict of generated report paths
    """
    generator = ReportGenerator(output_dir)
    return generator.generate(results, config)
