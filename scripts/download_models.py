import os
import urllib.request
import bz2
from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import snapshot_download


def setup_directories():
    dirs = [
        "models/weights",
        "models/face_library",
        "output",
        "scripts"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("Directory structure verified.")


def download_yolo_models():
    print("\n[1/4] Fetching YOLO Models...")
    
    print("Downloading YOLOv8n...")
    YOLO("yolov8n.pt")

    
    
    print("Downloading YOLOv8s baseline for event detection...")
    YOLO("yolov8s.pt")


def download_dlib_landmarks():
    print("\n[2/4] Fetching Dlib Facial Landmarks...")
    target_path = Path("models/weights/shape_predictor_68_face_landmarks.dat")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_path = target_path.with_suffix(".dat.bz2")

    if target_path.exists():
        print("Dlib landmarks already exist.")
        return

    print("Downloading shape_predictor_68_face_landmarks.dat.bz2 (approx. 64MB)...")
    try:
        urllib.request.urlretrieve(url, compressed_path)
        print("Decompressing bz2 file...")
        with bz2.BZ2File(compressed_path) as fr, open(target_path, "wb") as fw:
            fw.write(fr.read())
        os.remove(compressed_path)
        print("Dlib landmarks ready.")
    except Exception as e:
        print(f"Failed to download Dlib landmarks: {e}")


def download_vlm_moondream():
    print("\n[3/4] Fetching Moondream2 VLM from Hugging Face...")
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26"
    try:
        snapshot_download(repo_id=model_id, revision=revision)
        print("Moondream2 weights cached successfully.")
    except Exception as e:
        print(f"Failed to download VLM weights: {e}")


def download_kinetics_weights():
    print("\n[4/4] Pre-caching Kinetics-400 R3D18 weights...")
    import torch
    try:
        torch.hub.load_state_dict_from_url(
            "https://download.pytorch.org/models/r3d_18-b3b3357e.pth",
            model_dir="models/weights"
        )
        print("Kinetics-400 R3D18 weights cached.")
    except Exception as e:
        print(f"Failed to cache Kinetics weights: {e}")


if __name__ == "__main__":
    setup_directories()
    download_yolo_models()
    download_dlib_landmarks()
    download_vlm_moondream()
    download_kinetics_weights()
    print("\nAll external foundational model downloads complete.")
