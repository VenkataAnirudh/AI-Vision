# AI Vision: Your World, Our Watch 🛡️

**AI Vision** is a production-grade, state-of-the-art surveillance analytics platform. Think of it as a tireless, ultra-intelligent security guard that never blinks. It ingests video feeds and analyzes them in real-time using advanced neural networks to detect fires, track violent altercations, understand indoor behavioral context (like someone collapsing or standing in an aggressive posture), and even analyze audio for distress signals.

It's designed to filter out the noise and only alert you when genuine, verified threats occur—providing you with uncompromised safety and a flawless forensic intelligence dashboard.

---

## 🚀 How to Run the Platform

Running the AI Vision platform is incredibly simple. It operates via a robust backend API and a beautiful, interactive Streamlit dashboard.

### Step 1: Initial Setup (First Time Only)
To keep the environment clean and prevent package conflicts, you should set up a Python virtual environment and install the locked dependencies.

Open your terminal in the project folder and run:
```bash
python -m venv venv

venv\Scripts\activate

source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Launching the System
Once your environment is set up, you can launch both the backend processing engine and the frontend dashboard with a single click. 

Simply double-click the **`run.bat`** file in your folder, or run it from your activated terminal:
```bash
run.bat
```

**What this does:**
1. Boots up the **FastAPI Backend Server** on `http://localhost:8000` (which handles the heavy lifting, video processing, and neural network inference).
2. Launches the **Streamlit Intelligence Dashboard** on `http://localhost:8501` (your interactive UI).

Once the terminal confirms everything is running, open your web browser to the Streamlit URL. From there, you can upload surveillance feeds, toggle between LLM providers (GPT-4 / Gemini), and watch the telemetry unfold in real-time!
