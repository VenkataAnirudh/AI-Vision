import argparse
from pathlib import Path
from core.pipeline import VideoPipeline

def main():
    parser = argparse.ArgumentParser(description="Video Intelligence Pipeline")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--camera", type=int, help="Webcam ID for real-time mode", default=None)
    parser.add_argument("--stages", type=str, default="all", help="Comma-separated stages (e.g., person,face,emotion,event) or 'all'")
    
    args = parser.parse_args()
    
    if not args.video and args.camera is None:
        print("Error: Provide either --video <path> or --camera <id>")
        return

    stages_to_run = [s.strip() for s in args.stages.split(',')]
    
    pipeline = VideoPipeline(stages=stages_to_run)
    
    if args.video:
        if not Path(args.video).exists():
            print(f"File not found: {args.video}")
            return
        pipeline.process_video(args.video)
    else:
        pipeline.process_stream(args.camera)

if __name__ == "__main__":
    main()