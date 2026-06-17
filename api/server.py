"""
VisionAI — FastAPI Backend Server
──────────────────────────────────
Handles video upload, background processing, job status tracking,
WebSocket progress streaming, health monitoring, and report serving.

NOTE: Heavy imports (pipeline, torch, etc.) are LAZY — loaded only
when an analysis job is submitted, NOT at server startup. This ensures
the server boots fast and the /health endpoint is reachable immediately.
"""

import os
import sys
import uuid
import shutil
import time
import json
import asyncio
import threading
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


BASE_DIR = Path(__file__).resolve().parent.parent
os.chdir(str(BASE_DIR))  
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))  

from utils.debug_log import append_debug, check_imports, install_exception_hooks, log_exception

install_exception_hooks(source="api")
append_debug(f"API module loaded. cwd={Path.cwd()} python={sys.executable}", source="api")


app = FastAPI(
    title="VisionAI Intelligence API",
    description="Production-grade Video Intelligence Pipeline — Real-time surveillance analysis with multi-model fusion.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"


for d in [OUTPUT_DIR, OUTPUT_DIR / "reports", OUTPUT_DIR / "annotated",
          OUTPUT_DIR / "events", UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)


app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR), check_dir=False), name="output")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_jobs_lock = threading.Lock()
processing_jobs = {}


ws_connections = {}  

_start_time = time.time()

PIPELINE_RUNTIME_IMPORTS = [
    "supervision",
    "ultralytics",
    "cv2",
    "torch",
    "numpy",
    "yaml",
    "insightface",
    "deepface",
    "librosa",
    "moviepy",
    "fastapi",
    "streamlit",
]






@app.get("/health")
async def health_check():
    """Health check endpoint for startup readiness verification."""
    gpu_info = "N/A"
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            gpu_info = {
                "device": torch.cuda.get_device_name(0),
                "vram_total_mb": round(total / 1024 / 1024),
                "vram_free_mb": round(free / 1024 / 1024),
                "cuda_version": torch.version.cuda,
            }
    except Exception:
        pass

    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "gpu": gpu_info,
        "active_jobs": len([j for j in processing_jobs.values() if j.get("status") == "processing"]),
        "total_jobs": len(processing_jobs),
    }






def _update_job(job_id: str, updates: dict):
    """Thread-safe job status update."""
    with _jobs_lock:
        if job_id in processing_jobs:
            processing_jobs[job_id].update(updates)


def _notify_ws(job_id: str, message: dict):
    """Send progress message to all WebSocket connections for a job."""
    if job_id in ws_connections:
        dead = []
        for ws in ws_connections[job_id]:
            try:
                asyncio.run(ws.send_json(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_connections[job_id].remove(ws)


def run_analysis_task(job_id: str, video_path: str, llm_provider: str = "openai"):
    """Heavy lifting background task — runs the full pipeline.
    
    Pipeline and all heavy ML dependencies are imported HERE (lazily),
    not at server startup. This keeps the server boot fast.
    """
    append_debug(
        f"Job {job_id} starting. video={video_path} llm_provider={llm_provider} python={sys.executable}",
        source="api",
    )
    try:
        
        missing = check_imports(PIPELINE_RUNTIME_IMPORTS, source=f"job:{job_id}")
        if missing:
            hint = (
                "Missing Python modules before pipeline import: "
                + ", ".join(missing)
                + ". Run the app with venv\\Scripts\\python.exe and install requirements.txt in that environment."
            )
            append_debug(hint, source=f"job:{job_id}", level="ERROR")
            _update_job(job_id, {"status": "failed", "error": hint})
            _notify_ws(job_id, {"type": "error", "status": "failed", "error": hint})
            return

        from core.pipeline import VideoPipeline

        _update_job(job_id, {"status": "processing", "stage": "Loading pipeline..."})
        append_debug(f"Job {job_id} loading pipeline object.", source="api")

        pipeline = VideoPipeline(stages=['all'], llm_provider=llm_provider)

        def progress_cb(stage_name: str):
            append_debug(f"Job {job_id} stage: {stage_name}", source="pipeline")
            _update_job(job_id, {"stage": stage_name, "status": "processing"})
            _notify_ws(job_id, {"type": "progress", "stage": stage_name, "status": "processing"})

        result = pipeline.process_video(video_path, progress_callback=progress_cb)

        if result is None:
            append_debug(f"Job {job_id} failed: pipeline returned no result.", source="api", level="ERROR")
            _update_job(job_id, {"status": "failed", "error": "Pipeline returned no result."})
            return

        
        run_dir_name = result.get('run_dir_name', '')
        report_name = Path(result.get('json_path', '')).name

        _update_job(job_id, {
            "status": "completed",
            "filename": Path(video_path).name,
            "run_dir": run_dir_name,
            "report_url": f"/output/{run_dir_name}/reports/{report_name}",
            "video_url": f"/output/{run_dir_name}/annotated/annotated_{Path(video_path).name}",
            "heatmap_url": f"/output/{run_dir_name}/analytics/heatmap.png" if result.get('heatmap_path') else None,
            "trajectory_url": f"/output/{run_dir_name}/analytics/trajectories.png" if result.get('trajectory_path') else None,
            "pdf_url": f"/output/{run_dir_name}/reports/{Path(result.get('pdf_path', '')).name}" if result.get('pdf_path') else None,
            "log_url": f"/logs/{job_id}",
        })
        _notify_ws(job_id, {"type": "completed", "status": "completed"})
        append_debug(f"Job {job_id} completed successfully. result={result}", source="api")
        print(f"[API Backend] Job {job_id} completed successfully.")

    except Exception as e:
        import traceback
        log_exception(f"Job {job_id} failed", e, source="api")
        traceback.print_exc()
        _update_job(job_id, {"status": "failed", "error": str(e)})
        _notify_ws(job_id, {"type": "error", "status": "failed", "error": str(e)})


@app.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    llm_provider: str = Form("openai"),
):
    """
    Upload a video for analysis. Specify the LLM provider (openai or gemini).
    Returns a job_id for tracking progress.
    """
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / file.filename
    append_debug(
        f"Analyze request accepted. job={job_id} filename={file.filename} target={file_path} llm_provider={llm_provider}",
        source="api",
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    append_debug(f"Upload saved. job={job_id} bytes={file_path.stat().st_size}", source="api")

    with _jobs_lock:
        processing_jobs[job_id] = {
            "status": "queued",
            "filename": file.filename,
            "llm_provider": llm_provider,
            "stage": "Queued",
            "report_url": None,
            "video_url": None,
            "heatmap_url": None,
            "trajectory_url": None,
            "pdf_url": None,
            "log_url": None,
            "queued_at": datetime.now().isoformat(),
        }

    background_tasks.add_task(run_analysis_task, job_id, str(file_path), llm_provider)
    return {"job_id": job_id, "status": "queued", "llm_provider": llm_provider}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the current status of a processing job."""
    with _jobs_lock:
        job = processing_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/logs/{job_id}")
async def get_logs(job_id: str, lines: int = 400):
    """Return the tail of the run's per-run log file for a job."""
    with _jobs_lock:
        job = processing_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    run_dir_name = job.get("run_dir")
    if not run_dir_name:
        return {"log": "", "note": "Run directory not created yet."}

    logs_dir = OUTPUT_DIR / run_dir_name / "logs"
    if not logs_dir.exists():
        return {"log": "", "note": "No logs directory for this run."}

    log_files = sorted(logs_dir.glob("pipeline_*.log"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not log_files:
        return {"log": "", "note": "No log file found for this run."}

    log_file = log_files[0]
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        tail = "".join(all_lines[-int(lines):])
        return {"log": tail, "file": log_file.name, "total_lines": len(all_lines)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read log: {e}")






@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """Real-time progress streaming via WebSocket."""
    await websocket.accept()

    if job_id not in ws_connections:
        ws_connections[job_id] = []
    ws_connections[job_id].append(websocket)

    try:
        while True:
            
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        if job_id in ws_connections:
            ws_connections[job_id] = [ws for ws in ws_connections[job_id] if ws != websocket]






@app.post("/query")
async def query_report_endpoint(job_id: str, question: str):
    """
    Submits a natural language query about a completed video analysis report.
    """
    with _jobs_lock:
        job = processing_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") != "completed":
        return {"answer": f"Analysis is currently in state: {job.get('status')}. Please wait until completed."}

    report_url = job.get("report_url")
    if not report_url:
        raise HTTPException(status_code=500, detail="Job marked completed but report URL is missing.")

    
    report_rel_path = report_url.lstrip("/")
    report_path = BASE_DIR / report_rel_path

    
    from features.nl_query import ReportQueryAgent

    llm_provider = job.get("llm_provider", "openai")
    agent = ReportQueryAgent(llm_provider_name=llm_provider)
    answer = agent.query_report(str(report_path), question)
    return {"answer": answer}






@app.get("/download/{job_id}/{file_type}")
async def download_file(job_id: str, file_type: str):
    """
    Download a specific output file from a completed job.
    file_type: 'json', 'pdf', 'srt', 'video', 'heatmap', 'trajectory'
    """
    with _jobs_lock:
        job = processing_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not yet completed")

    url_map = {
        "json": job.get("report_url"),
        "pdf": job.get("pdf_url"),
        "video": job.get("video_url"),
        "heatmap": job.get("heatmap_url"),
        "trajectory": job.get("trajectory_url"),
    }

    url = url_map.get(file_type)
    if not url:
        raise HTTPException(status_code=404, detail=f"File type '{file_type}' not available for this job.")

    file_path = BASE_DIR / url.lstrip("/")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk.")

    return FileResponse(str(file_path), filename=file_path.name)






@app.get("/runs")
async def list_past_runs():
    """List all past analysis runs with their output directories."""
    runs = []
    if OUTPUT_DIR.exists():
        for entry in sorted(OUTPUT_DIR.iterdir(), reverse=True):
            if entry.is_dir() and entry.name.startswith("run_"):
                report_files = list((entry / "reports").glob("*.json")) if (entry / "reports").exists() else []
                runs.append({
                    "run_dir": entry.name,
                    "has_report": len(report_files) > 0,
                    "created": datetime.fromtimestamp(entry.stat().st_ctime).isoformat(),
                })
    return {"runs": runs}






if __name__ == "__main__":
    import uvicorn
    append_debug(f"Starting server. host=0.0.0.0 port=8000 project_root={BASE_DIR}", source="api")
    print(f"[VisionAI] Starting server... Project root: {BASE_DIR}")
    print(f"[VisionAI] Output dir: {OUTPUT_DIR}")
    print(f"[VisionAI] Swagger docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
