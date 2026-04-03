<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 5f6d8b2f (Initial commit)
# 🎯 Multi-Object Detection & Persistent ID Tracking Pipeline

A fully automated computer vision pipeline for sports/event video analysis that detects persons frame-by-frame, assigns persistent unique IDs, and generates rich analytics.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **YOLOv8 Detection**: State-of-the-art real-time person detection
- **Kalman Filter Tracking**: Robust multi-object tracking with Hungarian algorithm
- **torchvision Re-Identification**: Persistent IDs across occlusions using MobileNet/ResNet
- **Rich Analytics**: Heatmaps, trajectories, team clustering, speed estimation
- **Professional UI**: Streamlit drag-and-drop interface
- **URL Support**: Download videos from YouTube using yt-dlp (requires installation)

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Modules](#modules)
- [Analytics](#analytics)
- [Limitations](#limitations)
- [Model Choices](#model-choices)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg installed on system

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd cv-tracking-pipeline
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Models (Automatic)

Models are downloaded automatically on first run:
- YOLOv8x weights (~130MB)
- OSNet weights (~18MB)

## ⚡ Quick Start

### Option 1: Streamlit UI (Recommended)

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Option 2: Command Line

```bash
python -m src.pipeline "path/to/video.mp4"
```

Or with a YouTube URL:

```bash
python -m src.pipeline "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Option 3: Python API

```python
from src.pipeline import TrackingPipeline

pipeline = TrackingPipeline(config_path="config.yaml")
results = pipeline.run("video.mp4")

print(f"Output: {results['annotated_video']}")
print(f"Tracks: {results['tracking_metrics']['total_tracks']}")
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
│              Drag-and-Drop UI / Video URL / File                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                       PREPROCESSOR                               │
│                 yt-dlp download + ffmpeg frames                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DETECTION                                 │
│                    YOLOv8x / YOLOv9e                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│   ByteTrack     │     │   BoT-SORT      │
│ (crowded scenes)│     │ (camera motion) │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RE-IDENTIFICATION                             │
│              OSNet embeddings + cosine similarity               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    ▼                 ▼                 ▼
┌─────────┐     ┌───────────┐     ┌───────────┐
│Analytics│     │   Team    │     │ Bird's-Eye│
│heatmap  │     │ Clustering│     │   View    │
│speed    │     │  KMeans   │     │homography │
└────┬────┘     └─────┬─────┘     └─────┬─────┘
     │                │                 │
     └────────────────┴─────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT PACKAGE                              │
│        annotated video + stats JSON + report + heatmap          │
└─────────────────────────────────────────────────────────────────┘
```

## ⚙️ Configuration

All parameters are configurable via `config.yaml`:

```yaml
# Detection
detection:
  model: "yolov8x.pt"
  confidence_threshold: 0.35
  nms_iou_threshold: 0.45

# Tracking
tracking:
  primary_tracker: "bytetrack"
  secondary_tracker: "botsort"
  ensemble:
    enabled: true

# Re-Identification
reid:
  enabled: true
  model: "osnet_x1_0"
  similarity_threshold: 0.65

# Analytics
analytics:
  heatmap:
    enabled: true
  team_clustering:
    enabled: true
    num_teams: 2
```

See `config.yaml` for all options.

## 📦 Modules

### Detector (`src/detector.py`)

YOLOv8 person detection with:
- Configurable confidence threshold
- NMS IoU threshold
- FP16 inference for speed
- Batch processing support

### Tracker (`src/tracker.py`)

Kalman Filter tracking with:
- **Kalman Filter**: Predicts object positions between frames
- **Hungarian Algorithm**: Optimal detection-to-track assignment
- **IoU Matching**: Handles crowded scenes effectively

### ReID (`src/reid.py`)

torchvision-based re-identification:
- MobileNetV3/ResNet appearance embeddings
- Gallery management per track
- Cosine similarity matching
- Handles occlusions and re-entries

### Annotator (`src/annotator.py`)

Visualization with OpenCV:
- Bounding boxes with track IDs
- Confidence score overlay
- Trajectory traces
- Team-based color coding

### Analytics (`src/analytics.py`)

Full enhancement suite:
- Movement heatmaps
- Bird's-eye view projection
- Team clustering (jersey color)
- Speed estimation
- Object count over time
- Evaluation metrics

## 📊 Analytics Features

### Movement Heatmap
Accumulates player positions and applies Gaussian smoothing to show activity density.

### Bird's-Eye View
Projects player positions to a 2D pitch map using homography transformation.

### Team Clustering
Extracts HSV color histograms from player crops and uses KMeans to separate teams.

### Speed Estimation
Calculates pixel-per-frame displacement, converts to m/s using calibration factor.

### Evaluation Metrics
- Total tracks
- Average track length
- ID switches
- Fragmentation rate

## ⚠️ Limitations

1. **Extended Occlusions**: Complete occlusion for >30 frames may cause ID loss
2. **Similar Appearance**: Difficulty distinguishing players with very similar appearances
3. **Camera Cuts**: Abrupt scene changes can break tracking continuity
4. **Crowded Scenes**: Performance degrades with >20 overlapping persons
5. **Lighting**: Poor lighting affects appearance-based re-identification

## 🧠 Model Choices

### Why YOLOv8?
- Best real-time multi-object detector
- Excellent accuracy/speed tradeoff
- Native person class support
- Active community and updates

### Why Kalman Filter + Hungarian Algorithm?
- Pure Python/NumPy implementation (no external dependencies)
- Proven effective for multi-object tracking
- Handles temporary occlusions well
- Computationally efficient

### Why torchvision ReID?
- Uses pretrained MobileNetV3/ResNet models
- Good generalization to sports scenarios
- Fast inference suitable for real-time
- No additional package dependencies

## 📁 Output Structure

```
outputs/
├── {video_name}/
│   ├── {video_name}_annotated.mp4  # Annotated video
│   ├── heatmap.png                  # Movement heatmap
│   ├── count_plot.png               # Object count over time
│   ├── metrics.json                 # Tracking metrics
│   ├── speed_stats.json             # Per-track speed data
│   ├── config_used.json             # Configuration used
│   ├── report.md                    # Technical report
│   └── sample_frame_*.jpg           # Sample frames
```

## 🔧 Troubleshooting

### CUDA Out of Memory
Reduce `imgsz` in config.yaml or use a smaller model (yolov8l/m/s).

### Slow Processing
- Enable frame_skip (e.g., 2-3)
- Reduce max_dimension
- Use FP16 inference (enabled by default)

### Poor Tracking
- Increase confidence_threshold for cleaner detections
- Adjust similarity_threshold for ReID
- Enable ensemble tracking

## 📜 License

MIT License - feel free to use for academic and commercial purposes.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [boxmot](https://github.com/mikel-brostrom/boxmot) for tracking algorithms
- [torchreid](https://github.com/KaiyangZhou/deep-person-reid) for OSNet
- [supervision](https://github.com/roboflow/supervision) for annotation
<<<<<<< HEAD
=======
# AI--Project
>>>>>>> e0b0db39a68341238b0f7c04c917e18347cc4967
=======
>>>>>>> 5f6d8b2f (Initial commit)
