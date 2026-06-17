import argparse
import sys
from pathlib import Path
from utils.debug_log import append_debug, check_imports, install_exception_hooks


PIPELINE_RUNTIME_IMPORTS = [
    "supervision",
    "ultralytics",
    "cv2",
    "torch",
    "numpy",
    "yaml",
]

def main():
    install_exception_hooks(source="main")
    append_debug(f"CLI started. argv={sys.argv} python={sys.executable}", source="main")
    parser = argparse.ArgumentParser(description="Video Intelligence Pipeline")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--camera", type=int, help="Webcam ID for real-time mode", default=None)
    parser.add_argument("--stages", type=str, default="all", help="Comma-separated stages (e.g., person,face,emotion,event) or 'all'")
    
    args = parser.parse_args()
    
    if not args.video and args.camera is None:
        print("Error: Provide either --video <path> or --camera <id>")
        return

    stages_to_run = [s.strip() for s in args.stages.split(',')]
    missing = check_imports(PIPELINE_RUNTIME_IMPORTS, source="main")
    if missing:
        print("Missing Python modules: " + ", ".join(missing))
        print("Run: venv\\Scripts\\python.exe -m pip install -r requirements.txt")
        return

    from core.pipeline import VideoPipeline
    
    pipeline = VideoPipeline(stages=stages_to_run)
    
    if args.video:
        if not Path(args.video).exists():
            append_debug(f"Input video not found: {args.video}", source="main", level="ERROR")
            print(f"File not found: {args.video}")
            return
        pipeline.process_video(args.video)
    else:
        pipeline.process_stream(args.camera)

if __name__ == "__main__":
    main()
