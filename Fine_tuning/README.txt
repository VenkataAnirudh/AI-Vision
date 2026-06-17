================================================================================
 VisionAI — FINE-TUNING SUITE
 Revamping every weak detector (everything except fire) with domain-matched data
================================================================================

WHY THIS EXISTS
---------------
The deployed pipeline only has ONE genuinely strong detector: fire/smoke
(firedetect-11s.pt). Everything else is either a weak pretrained model or a
hand-tuned heuristic that false-fires on normal home footage. This suite
fine-tunes 5 models on domain-matched data + hard negatives so they actually
work in a home setting.

  Folder                  Model              Replaces / fixes
  ----------------------  -----------------  ------------------------------------
  01_violence_stgcn       ST-GCN (skeleton)  VideoMAE + aggressive_guard heuristic
  02_home_action_stgcn    ST-GCN (skeleton)  dual-head R3D (indoor_action)
  03_weapon_yolo          YOLO11n (detect)   domain-biased pretrained weapon YOLO
  04_emotion_fer          CNN (image)        DeepFace emotion + brow/EAR heuristics
  05_audio_events         CNN (log-mel)      audio_analysis.py DSP cry rules

Fire detection is intentionally NOT here — it works, leave it alone.


HARDWARE / TIME CONSTRAINTS (these shaped every design choice)
--------------------------------------------------------------
* TRAINING runs on Google Colab (free T4, 16 GB). Each notebook is built to
  finish inside ONE ~4 hour session. The slow step (skeleton / spectrogram
  extraction) is CACHED to Drive and is RESUME-SAFE — if Colab disconnects,
  just re-run; it skips what's already done.

* INFERENCE runs on your local 4 GB GTX 1650 Ti. So every model here is
  deliberately LIGHTWEIGHT:
    - skeleton models default to yolo11s-pose (NOT yolo11x-pose) for pose
      extraction — lighter on 4 GB AND ~4x faster extraction on Colab.
      >>> IMPORTANT: set the SAME pose model in the app's config.yaml
          (models.indoor_action.model_path) so train/inference match. <<<
    - weapon = YOLO11n (smallest)
    - emotion = EfficientNet-B0 / MobileNetV3 (tiny)
    - audio = small log-mel CNN

* RUN THEM IN PARALLEL: each folder is fully independent. Open all 5 notebooks
  in separate Colab runtimes (use separate Google accounts if you want true
  parallel GPUs on free tier) and run them at the same time.


HOW YOU'LL ACTUALLY USE THIS (your stated workflow)
---------------------------------------------------
1. On your LOCAL PC (PowerShell), download each dataset using the commands in
   that folder's DATASETS.txt. Every command downloads DIRECTLY into the right
   model folder. Kaggle CLI works because AnamolyD/kaggle.json already has your
   credentials (copy it to %USERPROFILE%\.kaggle\kaggle.json).
2. Upload the WHOLE Fine_tuning folder to Google Drive at:
        MyDrive/VisionAI/Fine_tuning/
   The notebook CONFIG cells are ALREADY pointed at that path, so there's
   nothing to edit (unless you skip an optional source - then set it to "").
3. Open the matching *_colab.ipynb in Colab, mount Drive, Runtime > Run all.
4. Each notebook saves its checkpoint to <model_folder>/_work on your Drive.
   Download it into the repo's models/weights/ (exact filename listed below).
5. Tell me when a checkpoint is downloaded and I'll wire its inference stage
   into the repo (the heuristic/weak model it replaces gets removed then).


OUTPUT CHECKPOINTS (what each notebook produces)
------------------------------------------------
  01  ->  violence_stgcn_best.pt
  02  ->  home_action_stgcn_best.pt
  03  ->  weapon_yolo11n_best.pt
  04  ->  emotion_fer_best.pt
  05  ->  audio_event_cnn_best.pt

Each checkpoint embeds its own config (class names, input sizes, pose model,
normalization) so the inference stage needs no guesswork.


ACCURACY LEVERS BAKED INTO EVERY NOTEBOOK
-----------------------------------------
* Hard negatives from the SAME domain (home footage) — the single biggest fix
  for false alarms, which was the previous models' failure mode.
* Class weighting + a weighted sampler for imbalanced classes.
* Strong but cheap augmentation.
* Train/inference parity (same pose model, same preprocessing).
* Precision-first evaluation (home = punish false alarms): every notebook
  prints per-class precision/recall + a confusion matrix, and the inference
  stages apply a confidence threshold + temporal persistence before alerting.

Start with the DATASETS.txt in each folder — downloading the data is the long
pole, so kick those downloads off first.
