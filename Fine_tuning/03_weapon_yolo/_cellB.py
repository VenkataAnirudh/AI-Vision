# === ADD-ON CELL B : CV summary + export fold0 as the deployment model ===
import os, json, shutil
import numpy as np
from ultralytics import YOLO

# Deploy fold0 (confirmed: best.pt + last.pt exist in cv_folds/fold0_weights/).
# Set to None to instead auto-pick the highest-mAP fold that has a saved ckpt.
DEPLOY_CKPT_OVERRIDE = WORK_DIR + "/cv_folds/fold0_weights/best.pt"

RESULTS_JSON = os.path.join(WORK_DIR, "cv_results.json")
results = json.load(open(RESULTS_JSON))
done = [results[k] for k in sorted(results)]
maps = [r["map"] for r in done]
print("CV mAP50-95 per fold:", {r["fold"]: round(r["map"], 4) for r in done})
print("CV mean=%.4f  std=%.4f  (n=%d)" % (float(np.mean(maps)), float(np.std(maps)), len(maps)))

have_ckpt = [r for r in done if r.get("ckpt") and os.path.isfile(r["ckpt"])]
sel = None
if DEPLOY_CKPT_OVERRIDE and os.path.isfile(DEPLOY_CKPT_OVERRIDE):
    sel_ckpt = DEPLOY_CKPT_OVERRIDE
    print("Deploying override ckpt:", sel_ckpt)
elif have_ckpt:
    sel = max(have_ckpt, key=lambda r: r["map"]); sel_ckpt = sel["ckpt"]
    print("Deploying highest-mAP fold with saved weights: fold %d (mAP=%.4f)" % (sel["fold"], sel["map"]))
else:
    raise SystemExit("No saved checkpoint found and override path missing.")

m = YOLO(sel_ckpt)
if sel and sel.get("data") and os.path.isfile(sel["data"]):
    metrics = m.val(data=sel["data"], imgsz=IMGSZ, device=0)
    print("mAP50-95:", round(float(metrics.box.map), 4), "| mAP50:", round(float(metrics.box.map50), 4))
    try:
        for i, name in m.names.items():
            print("  %-14s mAP50=%.3f" % (name, float(metrics.box.maps[i])))
    except Exception as e:
        print("per-class map unavailable:", e)

export_path = os.path.join(WORK_DIR, "weapon_yolo11n_best.pt")
shutil.copy(sel_ckpt, export_path)
print("Exported ->", export_path)
