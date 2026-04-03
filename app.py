"""
Streamlit Application - Professional Video Tracking UI
======================================================

A beautiful drag-and-drop interface for the CV tracking pipeline.
"""

import streamlit as st
import os
import sys
import time
from pathlib import Path
import tempfile
import json
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import TrackingPipeline, PipelineConfig

# Page configuration
st.set_page_config(
    page_title="Sports Video Tracking Pipeline",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Upload area styling */
    .upload-container {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #764ba2;
        background: linear-gradient(180deg, #fff 0%, #f8f9fa 100%);
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .feature-card h4 {
        color: #2d3436;
        margin: 0 0 0.5rem 0;
    }
    
    .feature-card p {
        color: #636e72;
        margin: 0;
        font-size: 0.9rem;
    }
    
    /* Progress section */
    .progress-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
    }
    
    /* Status indicators */
    .status-success {
        color: #00b894;
        font-weight: 600;
    }
    
    .status-processing {
        color: #0984e3;
        font-weight: 600;
    }
    
    .status-error {
        color: #d63031;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2d3436 0%, #636e72 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Video container */
    .video-container {
        background: #000;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* Results section */
    .results-header {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.5rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f1f3f5;
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animations */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .processing-indicator {
        animation: pulse 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Sports Video Tracking Pipeline</h1>
        <p>Multi-Object Detection & Persistent ID Tracking for Sports/Event Footage</p>
    </div>
    """, unsafe_allow_html=True)


def render_features():
    """Render feature highlights."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🔍 YOLOv8/v9 Detection</h4>
            <p>State-of-the-art person detection with configurable confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Ensemble Tracking</h4>
            <p>ByteTrack + BoT-SORT for robust ID persistence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>👤 OSNet ReID</h4>
            <p>Re-identification across occlusions and camera cuts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Rich Analytics</h4>
            <p>Heatmaps, trajectories, team clustering, speed stats</p>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with configuration options."""
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Detection settings
    st.sidebar.markdown("### 🔍 Detection")
    detection_model = st.sidebar.selectbox(
        "Model",
        ["yolov8m.pt", "yolov8s.pt", "yolov8n.pt", "yolov8l.pt", "yolov8x.pt", "yolov9c.pt", "yolov9e.pt"],
        index=0,
        help="YOLOv8m is balanced, YOLOv8n is fastest, YOLOv8x is most accurate"
    )
    
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.35,
        step=0.05,
        help="Minimum detection confidence"
    )
    
    nms_iou = st.sidebar.slider(
        "NMS IoU Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        help="Non-maximum suppression threshold"
    )
    
    # Tracking settings
    st.sidebar.markdown("### 🎯 Tracking")
    primary_tracker = st.sidebar.selectbox(
        "Primary Tracker",
        ["kalman", "simple"],
        index=0,
        help="Kalman filter for smooth tracking"
    )
    
    ensemble_enabled = st.sidebar.checkbox(
        "Enable Ensemble Tracking",
        value=False,
        help="Not needed with Kalman tracker"
    )
    
    # ReID settings
    st.sidebar.markdown("### 👤 Re-Identification")
    reid_enabled = st.sidebar.checkbox(
        "Enable ReID",
        value=True,
        help="Use MobileNet for persistent ID tracking"
    )
    
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.3,
        max_value=0.9,
        value=0.60,
        step=0.05,
        help="Threshold for re-identification matching"
    )
    
    # Analytics settings
    st.sidebar.markdown("### 📊 Analytics")
    enable_heatmap = st.sidebar.checkbox("Generate Heatmap", value=True)
    enable_team_clustering = st.sidebar.checkbox("Team Clustering", value=True)
    enable_birds_eye = st.sidebar.checkbox("Bird's-Eye View", value=False)
    enable_speed = st.sidebar.checkbox("Speed Estimation", value=True)
    
    # Processing settings
    st.sidebar.markdown("### 🎬 Processing")
    frame_skip = st.sidebar.slider(
        "Frame Skip",
        min_value=1,
        max_value=10,
        value=2,
        help="Process every Nth frame (higher = faster)"
    )
    
    max_dimension = st.sidebar.slider(
        "Max Resolution",
        min_value=480,
        max_value=1920,
        value=960,
        step=80,
        help="Resize frames if larger"
    )
    
    # Model comparison info
    st.sidebar.markdown("### 📊 Model Comparison")
    st.sidebar.markdown("""
    **Detection Models:**
    | Model | Speed | Accuracy |
    |-------|-------|----------|
    | YOLOv8n | ⚡⚡⚡ | ⭐⭐ |
    | YOLOv8s | ⚡⚡ | ⭐⭐⭐ |
    | YOLOv8m | ⚡⚡ | ⭐⭐⭐⭐ |
    | YOLOv8l | ⚡ | ⭐⭐⭐⭐ |
    | YOLOv8x | ⚡ | ⭐⭐⭐⭐⭐ |
    
    *Recommended: YOLOv8m for balanced performance*
    """)
    
    return {
        'detection_model': detection_model,
        'confidence_threshold': confidence,
        'nms_iou_threshold': nms_iou,
        'primary_tracker': primary_tracker,
        'secondary_tracker': 'botsort' if primary_tracker == 'bytetrack' else 'bytetrack',
        'ensemble_enabled': ensemble_enabled,
        'reid_enabled': reid_enabled,
        'similarity_threshold': similarity_threshold,
        'enable_heatmap': enable_heatmap,
        'enable_team_clustering': enable_team_clustering,
        'enable_birds_eye': enable_birds_eye,
        'enable_speed': enable_speed,
        'frame_skip': frame_skip,
        'max_dimension': max_dimension
    }


def render_upload_section():
    """Render the video upload section."""
    st.markdown("## 📤 Upload Video")
    
    tab1, tab2 = st.tabs(["📁 Upload File", "🔗 Video URL"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Drag and drop your video file here",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Supported formats: MP4, AVI, MOV, MKV, WebM"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")
            return ('file', uploaded_file)
    
    with tab2:
        url = st.text_input(
            "Enter video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a YouTube or direct video URL"
        )
        
        if url:
            st.info(f"🔗 URL provided: {url}")
            return ('url', url)
    
    return (None, None)


def render_results(results: dict):
    """Render the results section."""
    st.markdown("""
    <div class="results-header">
        <h2 style="margin: 0;">✅ Processing Complete!</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Source info
    source_url = results.get('source_url', 'Unknown')
    st.info(f"📹 **Source:** {source_url}")
    
    # Metrics cards
    stats = results.get('statistics', {})
    metrics = results.get('tracking_metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Frames",
            stats.get('total_frames', 'N/A'),
            help="Number of frames processed"
        )
    
    with col2:
        st.metric(
            "Processing FPS",
            f"{stats.get('avg_fps', 0):.1f}",
            help="Average processing speed"
        )
    
    with col3:
        st.metric(
            "Total Tracks",
            metrics.get('total_tracks', 'N/A'),
            help="Unique persons tracked"
        )
    
    with col4:
        st.metric(
            "Avg Track Length",
            f"{metrics.get('avg_track_length', 0):.0f}",
            help="Average frames per track"
        )
    
    # Tabs for different outputs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎬 Video", "🗺️ Heatmap", "📈 Analytics", 
        "👥 Teams", "🏃 Speed", "📄 Report"
    ])
    
    with tab1:
        st.markdown("### Annotated Video")
        video_path = results.get('annotated_video')
        if video_path and os.path.exists(video_path):
            # Read video file as bytes for Streamlit
            with open(video_path, 'rb') as video_file:
                video_bytes = video_file.read()
            
            # Display video
            st.video(video_bytes)
            
            # Download button
            st.download_button(
                "⬇️ Download Video",
                video_bytes,
                file_name="annotated_video.mp4",
                mime="video/mp4"
            )
        else:
            st.warning("Annotated video not found. Check the outputs folder.")
    
    with tab2:
        st.markdown("### Movement Heatmap")
        heatmap_path = results.get('heatmap')
        if heatmap_path and os.path.exists(heatmap_path):
            st.image(heatmap_path, caption="Activity density heatmap - Brighter areas indicate more movement")
        else:
            st.info("Heatmap not available. Enable heatmap in settings.")
        
        # Bird's Eye View
        st.markdown("### 🏟️ Bird's Eye View")
        birds_eye_path = results.get('birds_eye_view')
        if birds_eye_path and os.path.exists(birds_eye_path):
            st.image(birds_eye_path, caption="Top-down pitch projection")
        else:
            st.info("Bird's eye view not available. This requires field calibration points.")
    
    with tab3:
        st.markdown("### 📊 Object Count Over Time")
        count_plot_path = results.get('count_plot')
        if count_plot_path and os.path.exists(count_plot_path):
            st.image(count_plot_path, caption="Number of tracked objects per frame")
        else:
            st.info("Count plot not available")
        
        # Show detailed metrics
        st.markdown("### 📋 Tracking Metrics")
        if metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Detections", metrics.get('total_detections', 'N/A'))
                st.metric("ID Switches", metrics.get('id_switches', 0))
                st.metric("Min Track Length", metrics.get('min_track_length', 'N/A'))
            with col2:
                st.metric("Avg Detections/Frame", f"{metrics.get('avg_detections_per_frame', 0):.1f}")
                st.metric("Fragmentations", metrics.get('fragmentations', 0))
                st.metric("Max Track Length", metrics.get('max_track_length', 'N/A'))
            
            # Full metrics table
            with st.expander("📊 View All Metrics"):
                metrics_df_data = {
                    'Metric': list(metrics.keys()),
                    'Value': [str(v) if not isinstance(v, float) else f"{v:.3f}" for v in metrics.values()]
                }
                st.table(metrics_df_data)
    
    with tab4:
        st.markdown("### 👥 Team Clustering Results")
        team_assignments = results.get('team_assignments', {})
        
        if team_assignments:
            st.success(f"✅ Successfully clustered {len(team_assignments)} players into teams!")
            
            # Count per team
            team_counts = {}
            for track_id, team_id in team_assignments.items():
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
            
            # Display team breakdown
            cols = st.columns(len(team_counts))
            team_colors = ["🔵 Team Blue", "🔴 Team Red", "🟢 Team Green"]
            
            for i, (col, (team_id, count)) in enumerate(zip(cols, team_counts.items())):
                with col:
                    team_name = team_colors[team_id] if team_id < len(team_colors) else f"Team {team_id}"
                    st.metric(team_name, f"{count} players")
            
            # Detailed assignments
            with st.expander("📋 View Player Assignments"):
                assignment_data = {
                    'Track ID': list(team_assignments.keys()),
                    'Team': [f"Team {v}" for v in team_assignments.values()]
                }
                st.table(assignment_data)
        else:
            st.info("Team clustering not available. Enable it in settings and ensure enough players are detected.")
    
    with tab5:
        st.markdown("### 🏃 Speed Statistics")
        speed_data = results.get('speed_data', {})
        
        if speed_data:
            st.success(f"✅ Speed data for {len(speed_data)} tracks")
            
            # Overall stats
            all_avg_speeds = [s['avg'] for s in speed_data.values() if s.get('avg', 0) > 0]
            all_max_speeds = [s['max'] for s in speed_data.values() if s.get('max', 0) > 0]
            
            if all_avg_speeds:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Speed", f"{sum(all_avg_speeds)/len(all_avg_speeds):.2f} m/s")
                with col2:
                    st.metric("Top Speed", f"{max(all_max_speeds):.2f} m/s")
                with col3:
                    # Convert to km/h for context
                    max_kmh = max(all_max_speeds) * 3.6
                    st.metric("Top Speed (km/h)", f"{max_kmh:.1f}")
            
            # Per-track breakdown
            with st.expander("📋 Per-Track Speed Data"):
                speed_table = []
                for track_id, stats in speed_data.items():
                    speed_table.append({
                        'Track ID': track_id,
                        'Avg Speed (m/s)': f"{stats.get('avg', 0):.2f}",
                        'Max Speed (m/s)': f"{stats.get('max', 0):.2f}",
                        'Max Speed (km/h)': f"{stats.get('max', 0) * 3.6:.1f}"
                    })
                if speed_table:
                    st.table(speed_table[:20])  # Show top 20
        else:
            st.info("Speed estimation not available. Enable it in settings.")
            st.caption("Note: Speed estimation requires pixels-per-meter calibration for accurate results.")
    
    with tab6:
        st.markdown("### 📄 Technical Report")
        
        # Check for report
        output_dir = results.get('output_dir', 'outputs')
        report_path = results.get('report') or os.path.join(output_dir, 'report.md')
        
        if report_path and os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            # Display report in expander
            with st.expander("📖 View Full Report", expanded=True):
                st.markdown(report_content)
            
            st.download_button(
                "⬇️ Download Report (Markdown)",
                report_content,
                file_name="TECHNICAL_REPORT.md",
                mime="text/markdown"
            )
        else:
            st.warning("Report not generated. Check output directory.")
    
    # Sample frames gallery
    st.markdown("---")
    st.markdown("### 📸 Sample Frames")
    sample_frames = results.get('sample_frames', [])
    if sample_frames:
        cols = st.columns(min(len(sample_frames), 5))
        for i, (col, frame_path) in enumerate(zip(cols, sample_frames[:5])):
            if os.path.exists(frame_path):
                col.image(frame_path, caption=f"Frame {i+1}")
    else:
        st.info("No sample frames available")
    
    # Download all outputs
    st.markdown("---")
    st.markdown("### 📦 Download All Outputs")
    
    output_dir = results.get('output_dir')
    if output_dir and os.path.exists(output_dir):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Metrics JSON
            metrics_path = results.get('metrics')
            if metrics_path and os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    st.download_button(
                        "📊 Download Metrics (JSON)",
                        f.read(),
                        file_name="metrics.json",
                        mime="application/json"
                    )
        
        with col2:
            # Speed stats JSON
            speed_path = results.get('speed_stats')
            if speed_path and os.path.exists(speed_path):
                with open(speed_path, 'r') as f:
                    st.download_button(
                        "🏃 Download Speed Stats (JSON)",
                        f.read(),
                        file_name="speed_stats.json",
                        mime="application/json"
                    )
        
        with col3:
            # Config used
            config_path = os.path.join(output_dir, 'config_used.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    st.download_button(
                        "⚙️ Download Config (JSON)",
                        f.read(),
                        file_name="config_used.json",
                        mime="application/json"
                    )


def process_video(input_source: str, config: dict, progress_placeholder):
    """Process video with the pipeline."""
    # Create pipeline config
    pipeline_config = PipelineConfig(
        detection_model=config['detection_model'],
        confidence_threshold=config['confidence_threshold'],
        nms_iou_threshold=config['nms_iou_threshold'],
        primary_tracker=config['primary_tracker'],
        secondary_tracker=config['secondary_tracker'],
        ensemble_enabled=config['ensemble_enabled'],
        reid_enabled=config['reid_enabled'],
        similarity_threshold=config['similarity_threshold'],
        enable_heatmap=config['enable_heatmap'],
        enable_team_clustering=config['enable_team_clustering'],
        enable_birds_eye=config['enable_birds_eye'],
        enable_speed=config['enable_speed'],
        frame_skip=config['frame_skip'],
        max_dimension=config['max_dimension']
    )
    
    # Create pipeline
    pipeline = TrackingPipeline(config=pipeline_config)
    
    # Progress callback
    progress_bar = progress_placeholder.progress(0)
    status_text = progress_placeholder.empty()
    
    def update_progress(progress: float, message: str):
        progress_bar.progress(progress)
        status_text.text(message)
    
    try:
        # Run pipeline (report is generated inside pipeline now)
        results = pipeline.run(
            input_source,
            progress_callback=update_progress
        )
        
        return results
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        raise e
    
    finally:
        pipeline.cleanup()


def main():
    """Main application entry point."""
    render_header()
    render_features()
    
    # Sidebar configuration
    config = render_sidebar()
    
    st.markdown("---")
    
    # Upload section
    source_type, source_data = render_upload_section()
    
    # Process button
    if source_data is not None:
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_btn = st.button("🚀 Start Processing", use_container_width=True)
        
        if process_btn:
            # Prepare input
            if source_type == 'file':
                # Save uploaded file temporarily
                # Reset file pointer first
                source_data.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(source_data.read())
                    input_source = tmp.name
                    logger.info(f"Saved uploaded file to: {input_source}")
            else:
                input_source = source_data
            
            # Progress section
            st.markdown("---")
            st.markdown("## 🔄 Processing")
            
            progress_placeholder = st.container()
            
            with st.spinner("Initializing pipeline..."):
                try:
                    results = process_video(input_source, config, progress_placeholder)
                    
                    # Store results in session state
                    st.session_state['results'] = results
                    
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
            
            # Clean up temp file
            if source_type == 'file' and os.path.exists(input_source):
                os.unlink(input_source)
    
    # Show results if available
    if 'results' in st.session_state:
        st.markdown("---")
        render_results(st.session_state['results'])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 2rem;">
        <p>🎯 Multi-Object Detection & Tracking Pipeline</p>
        <p style="font-size: 0.8rem;">
            Powered by YOLOv8 • Kalman Filter Tracker • MobileNet ReID
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
