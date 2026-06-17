# === ADD-ON CELL A : run ONLY fold1 (fold0 already done). SAME configs. ===
# Run cells 1-7 first (drive mount, pip, normalize paths, optional negatives),
# then SKIP the original training cell (cell 9) and run THIS instead. It is
# self-contained: re-stages to local SSD (cheap if already staged), rebuilds the
# fold index, then trains ONLY the folds in FOLDS_TO_RUN, saving each fold's
# best.pt to Drive the moment it finishes. fold1 trains FRESH from yolo11n.pt
# (NOT warm-started from fold0 - that would leak fold0's val data into fold1).
import os, glob, shutil, json, time, yaml
import numpy as np
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO

FOLDS_TO_RUN = [1]        # fold0 already trained at 80 epochs; fold2 skipped
FOLD0_MAP    = 0.787      # used only for the CV mean, and only if fold0 isn't
                          # already recorded in cv_results.json

# ---- rebuild dataset index + fold builder (identical to the original cell) ----
cfg = yaml.safe_load(open(DATA_YAML))
ddir_drive = cfg.get("path", os.path.dirname(DATA_YAML))
names = cfg["names"] if isinstance(cfg["names"], list) else [cfg["names"][i] for i in sorted(cfg["names"])]
NC = len(names)
LOCAL_DS = "/content/weapon_ds"; os.makedirs(LOCAL_DS, exist_ok=True)
for field in (cfg.get("train"), cfg.get("val"), cfg.get("test")):
    if not field: continue
    top = field.replace("\\", "/").split("/")[0]
    src = os.path.join(ddir_drive, top); dst = os.path.join(LOCAL_DS, top)
    if os.path.isdir(src) and not os.path.isdir(dst):
        print("Staging %s -> local ..." % top); shutil.copytree(src, dst)
ddir = LOCAL_DS

def imgs_for(split_field):
    if not split_field: return []
    p = split_field if os.path.isabs(split_field) else os.path.join(ddir, split_field)
    if os.path.isdir(os.path.join(p, "images")): p = os.path.join(p, "images")
    out = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        out += glob.glob(os.path.join(p, "**", ext), recursive=True)
    return out

all_imgs = []
for f in (cfg.get("train"), cfg.get("val"), cfg.get("test")):
    all_imgs += imgs_for(f)
all_imgs = sorted(set(all_imgs))

def label_path(img):
    d = os.path.dirname(img).replace(os.sep + "images", os.sep + "labels")
    return os.path.join(d, os.path.splitext(os.path.basename(img))[0] + ".txt")

def strat_label(img):
    lp = label_path(img)
    if not os.path.isfile(lp): return NC
    cls = [int(float(ln.split()[0])) for ln in open(lp) if ln.strip()]
    return max(set(cls), key=cls.count) if cls else NC

all_imgs = np.array(all_imgs)
y_strat = np.array([strat_label(p) for p in all_imgs])
print("Images: %d" % len(all_imgs))

CV_ROOT = "/content/cv"
def build_fold_dir(fold, tr_idx, va_idx):
    root = os.path.join(CV_ROOT, "data", "fold%d" % fold)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = os.path.join(root, sub); os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, "*")): os.remove(f)
    def link(idxs, split):
        for i in idxs:
            img = str(all_imgs[i]); stem = os.path.splitext(os.path.basename(img))[0]
            dst_img = os.path.join(root, "images", split, os.path.basename(img))
            try: os.symlink(img, dst_img)
            except (OSError, NotImplementedError): shutil.copy(img, dst_img)
            lp = label_path(img)
            if os.path.isfile(lp):
                dst_lab = os.path.join(root, "labels", split, stem + ".txt")
                try: os.symlink(lp, dst_lab)
                except (OSError, NotImplementedError): shutil.copy(lp, dst_lab)
    link(tr_idx, "train"); link(va_idx, "val")
    yml = os.path.join(root, "data.yaml")
    yaml.safe_dump({"path": root, "train": "images/train", "val": "images/val",
                    "nc": NC, "names": names}, open(yml, "w"))
    return yml

# ---- train only the requested folds; persist each best.pt to Drive ----
CV_SAVE = os.path.join(WORK_DIR, "cv_folds"); os.makedirs(CV_SAVE, exist_ok=True)
RESULTS_JSON = os.path.join(WORK_DIR, "cv_results.json")
results = json.load(open(RESULTS_JSON)) if os.path.isfile(RESULTS_JSON) else {}
results.setdefault("0", {"fold": 0, "map": FOLD0_MAP, "ckpt": None, "data": None})
os.makedirs(os.path.join(CV_ROOT, "runs"), exist_ok=True)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
for fold, (tr_idx, va_idx) in enumerate(skf.split(all_imgs, y_strat)):
    if fold not in FOLDS_TO_RUN:
        continue
    data_yaml = build_fold_dir(fold, tr_idx, va_idx)
    saved = os.path.join(CV_SAVE, "fold%d_best.pt" % fold)
    if str(fold) in results and results[str(fold)].get("ckpt") and os.path.isfile(saved):
        results[str(fold)]["data"] = data_yaml
        print("Fold %d already done (mAP=%.4f) - skip" % (fold, results[str(fold)]["map"])); continue
    print("\n===== Fold %d/%d  train=%d val=%d =====" % (fold + 1, N_SPLITS, len(tr_idx), len(va_idx)))
    t0 = time.time()
    model = YOLO(MODEL)
    model.train(
        data=data_yaml, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
        patience=PATIENCE, seed=SEED, project=os.path.join(CV_ROOT, "runs"),
        name="fold%d" % fold, exist_ok=True, pretrained=True, optimizer="auto", cos_lr=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, mosaic=1.0,
        device=0, verbose=True)
    best_pt = os.path.join(CV_ROOT, "runs", "fold%d" % fold, "weights", "best.pt")
    vm = YOLO(best_pt).val(data=data_yaml, imgsz=IMGSZ, device=0, verbose=False)
    map95 = float(vm.box.map)
    shutil.copy(best_pt, saved)
    results[str(fold)] = {"fold": fold, "map": map95, "ckpt": saved, "data": data_yaml}
    json.dump(results, open(RESULTS_JSON, "w"))
    print("Fold %d done in %.1f min | mAP50-95=%.4f -> %s" % (fold, (time.time()-t0)/60, map95, saved))

done = [results[str(f)] for f in range(N_SPLITS) if str(f) in results]
maps = [r["map"] for r in done]
print("\nCV mAP50-95 per fold:", {r["fold"]: round(r["map"], 4) for r in done},
      "| mean=%.4f +/- %.4f" % (float(np.mean(maps)), float(np.std(maps))))
