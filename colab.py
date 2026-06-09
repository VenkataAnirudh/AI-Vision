# Indoor Action Recognition Fine-tuning on Google Colab
# ----------------------------------------------------
# This script is designed for Google Colab to:
# 1. Mount Google Drive
# 2. Download/install all dependencies
# 3. Set up dataset and code paths
# 4. Run the fine-tuning script
# 5. Save trained weights back to Google Drive
#
# Edit the CONFIGURABLE PARAMETERS section as needed.

# =========================
# CONFIGURABLE PARAMETERS
# =========================
# Path to your dataset in Google Drive (change as needed)
DATASET_PATH = "/content/drive/MyDrive/IndoorActionDataset-video"
# Path to save weights in Google Drive
WEIGHTS_SAVE_PATH = "/content/drive/MyDrive/indoor_action_r3d18.pt"
# Number of epochs
EPOCHS = 25
# Batch size
BATCH_SIZE = 4
# Learning rate
LEARNING_RATE = 3e-4

# =========================
# 1. Mount Google Drive
# =========================
from google.colab import drive
drive.mount('/content/drive')

# =========================
# 2. Install Dependencies
# =========================
# (Uncomment and run these lines manually in a Colab cell if needed)
# !pip install torch torchvision opencv-python pyyaml

# =========================
# 3. Clone/Copy Your Codebase
# =========================
# If your code is in Google Drive, copy it to /content for faster access:
import shutil
import os

CODE_SRC = "/content/drive/MyDrive/your_project_folder"  # CHANGE THIS to your code folder in Drive
CODE_DST = "/content/indoor_action_code"

if not os.path.exists(CODE_DST):
    shutil.copytree(CODE_SRC, CODE_DST)

os.chdir(CODE_DST)

# =========================
# 4. Symlink or Copy Dataset
# =========================
if not os.path.exists("IndoorActionDataset-video"):
    os.symlink(DATASET_PATH, "IndoorActionDataset-video")

# =========================
# 5. Run Training Script
# =========================
import subprocess
subprocess.run([
    "python", "scripts/train_indoor_action.py",
    "--epochs", str(EPOCHS),
    "--batch_size", str(BATCH_SIZE),
    "--lr", str(LEARNING_RATE)
])

# =========================
# 6. Save Trained Weights to Drive
# =========================
import shutil
if os.path.exists("models/weights/indoor_action_r3d18.pt"):
    shutil.copy("models/weights/indoor_action_r3d18.pt", WEIGHTS_SAVE_PATH)
    print(f"Saved trained weights to {WEIGHTS_SAVE_PATH}")
else:
    print("Trained weights not found!")

# =========================
# 7. Download Weights to Local (Optional)
# =========================
# from google.colab import files
# files.download(WEIGHTS_SAVE_PATH)
