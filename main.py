"""
Main entry point for the CV Tracking Pipeline.
Run with: python main.py <video_path_or_url>
"""

import sys
import argparse
from pathlib import Path

from src.pipeline import TrackingPipeline, PipelineConfig
from src.report import ReportGenerator
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Object Detection & Persistent ID Tracking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py video.mp4
  python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
  python main.py video.mp4 --config custom_config.yaml
  python main.py video.mp4 --model yolov8l.pt --confidence 0.4
        """
    )
    
    parser.add_argument(
        "input",
        help="Video file path or URL (YouTube, etc.)"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output name (default: derived from input)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Detection model (overrides config)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Detection confidence threshold (overrides config)"
    )
    
    parser.add_argument(
        "--no-reid",
        action="store_true",
        help="Disable re-identification"
    )
    
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable ensemble tracking"
    )
    
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        help="Process every Nth frame (overrides config)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps"],
        default=None,
        help="Compute device (overrides config)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = PipelineConfig.from_yaml(str(config_path))
        logger.info(f"Loaded config from: {config_path}")
    else:
        config = PipelineConfig()
        logger.warning(f"Config file not found: {config_path}, using defaults")
    
    # Apply command-line overrides
    if args.model:
        config.detection_model = args.model
    if args.confidence:
        config.confidence_threshold = args.confidence
    if args.no_reid:
        config.reid_enabled = False
    if args.no_ensemble:
        config.ensemble_enabled = False
    if args.frame_skip:
        config.frame_skip = args.frame_skip
    if args.device:
        config.device = args.device
    
    # Create pipeline
    logger.info("Initializing pipeline...")
    pipeline = TrackingPipeline(config=config)
    
    # Progress callback
    def progress_callback(progress: float, message: str):
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r[{bar}] {progress*100:.1f}% - {message}", end='', flush=True)
        if progress >= 1.0:
            print()
    
    # Run pipeline
    try:
        results = pipeline.run(
            args.input,
            output_name=args.output,
            progress_callback=progress_callback
        )
        
        # Generate report
        report_gen = ReportGenerator(results['output_dir'])
        report_gen.generate(
            results,
            config.__dict__,
            formats=['markdown']
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("              PROCESSING COMPLETE")
        print("=" * 60)
        print(f"\n📁 Output Directory: {results['output_dir']}")
        print(f"🎬 Annotated Video: {results['annotated_video']}")
        print(f"\n📊 Statistics:")
        
        stats = results.get('statistics', {})
        print(f"   • Total Frames: {stats.get('total_frames', 'N/A')}")
        print(f"   • Processing Time: {stats.get('total_time_seconds', 0):.1f}s")
        print(f"   • Average FPS: {stats.get('avg_fps', 0):.1f}")
        
        metrics = results.get('tracking_metrics', {})
        if metrics:
            print(f"\n🎯 Tracking Metrics:")
            print(f"   • Total Tracks: {metrics.get('total_tracks', 'N/A')}")
            print(f"   • Avg Track Length: {metrics.get('avg_track_length', 0):.1f} frames")
            print(f"   • ID Switches: {metrics.get('id_switches', 'N/A')}")
            print(f"   • Fragmentation Rate: {metrics.get('fragmentation_rate', 0):.3f}")
        
        print("\n" + "=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    sys.exit(main())
