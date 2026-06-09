import os
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from core.pipeline import VideoPipeline
from features.nl_query import ReportQueryAgent

app = FastAPI(title="Video Intelligence API")

# Configuration
BASE_DIR = Path("D:/Coding/Vision AI")
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Static Files (for serving processed videos/reports to UI)
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job tracker
processing_jobs = {}

def run_analysis_task(job_id: str, video_path: str):
    """Heavy lifting background task"""
    try:
        pipeline = VideoPipeline(stages=['all'])
        
        def progress_cb(stage_name: str):
            processing_jobs[job_id]["stage"] = stage_name
            processing_jobs[job_id]["status"] = "processing"
            
        report_path = pipeline.process_video(video_path, progress_callback=progress_cb)
        
        # Verify the report file name and local path
        report_name = Path(report_path).name
        
        processing_jobs[job_id] = {
            "status": "completed",
            "filename": Path(video_path).name,
            "report_url": f"/output/reports/{report_name}",
            "video_url": f"/output/annotated/annotated_{Path(video_path).name}"
        }
        print(f"[API Backend] Job {job_id} completed successfully. Report saved to {report_name}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        processing_jobs[job_id] = {"status": "failed", "error": str(e)}

@app.post("/analyze")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / file.filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    processing_jobs[job_id] = {
        "status": "processing",
        "filename": file.filename,
        "report_url": None,
        "video_url": None
    }
    background_tasks.add_task(run_analysis_task, job_id, str(file_path))
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = processing_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/query")
async def query_report_endpoint(job_id: str, question: str):
    """
    Submits a natural language query about a completed video analysis report.
    """
    job = processing_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job.get("status") != "completed":
        return {"answer": f"Analysis is currently in state: {job.get('status')}. Please wait until completed."}
        
    report_url = job.get("report_url")
    if not report_url:
        raise HTTPException(status_code=500, detail="Job marked completed but report URL is missing.")
        
    report_name = Path(report_url).name
    report_path = OUTPUT_DIR / "reports" / report_name
    
    agent = ReportQueryAgent()
    answer = agent.query_report(str(report_path), question)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)