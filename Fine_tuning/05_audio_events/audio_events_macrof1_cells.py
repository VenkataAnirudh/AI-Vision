# ============================================================================
#  NB 05 - ADD-ON CELLS  (paste these as TWO new cells at the BOTTOM of
#  audio_events_colab.ipynb, AFTER the existing Train + Evaluate cells)
#
#  WHAT THEY DO
#    Re-run the same Stratified K-Fold CV, but pick the best fold by MACRO-F1
#    (best selector for imbalanced cry/scream vs. 'other') instead of val_loss,
#    then save to a SEPARATE file: audio_event_cnn_best_macrof1.pt
#    Your original audio_event_cnn_best.pt is left untouched.
#
#  REQUIREMENTS
#    Runtime still connected from a full run. These reuse what's already in
#    memory: trainval_items, test_items, CLASS_NAMES, NUM_CLASSES, make_loaders,
#    AudioCNN, MelDataset, device, WORK_DIR and all hyperparams (SEED, N_SPLITS,
#    PATIENCE, EPOCHS, LR, WEIGHT_DECAY, BATCH_SIZE, SR, DURATION, ...).
#
#  COST
#    This retrains all folds again (~same time as the first training cell),
#    because per-fold weights were not kept in memory.
#
#  TO SELECT BY VAL-ACCURACY INSTEAD: change the metric in train_one_fold_f1
#    from  mf1 = f1_score(...)  to  mf1 = (y_true == y_pred).mean()
#    (and rename the save file). Macro-F1 is recommended though.
# ============================================================================


# === CELL A  -  Re-run CV, select best fold by MACRO-F1, save separately ===
import os, copy, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score

torch.manual_seed(SEED); np.random.seed(SEED)
macrof1_save_path = os.path.join(WORK_DIR, "audio_event_cnn_best_macrof1.pt")
# Alert classes = the distress sounds (NB 05 has no alert_idx in memory; derive it).
alert_idx = [i for i, c in enumerate(CLASS_NAMES) if c in ("baby_cry", "scream")]

def train_one_fold_f1(tr_items, va_items, fold):
    tr_loader, va_loader, cls_w = make_loaders(tr_items, va_items)
    model = AudioCNN(NUM_CLASSES).to(device)
    w = torch.tensor(cls_w / cls_w.sum() * NUM_CLASSES, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    best_f1 = -1.0; best_vloss = float("inf"); best_state = None; bad = 0
    for epoch in range(EPOCHS):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device); opt.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    out = model(x); loss = criterion(out, y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                out = model(x); loss = criterion(out, y); loss.backward(); opt.step()
        sched.step()
        # validation: collect preds for macro-F1 (primary) + val_loss (tie-break)
        model.eval(); vloss = 0.0; vn = 0; ys = []; ps = []
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(device), y.to(device)
                out = model(x); vloss += criterion(out, y).item() * y.size(0); vn += y.size(0)
                ps.append(out.argmax(1).cpu().numpy()); ys.append(y.cpu().numpy())
        y_true = np.concatenate(ys); y_pred = np.concatenate(ps); vloss /= max(vn, 1)
        mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        arec = recall_score(y_true, y_pred, labels=alert_idx, average="macro",
                            zero_division=0) if alert_idx else 0.0
        print("  [fold %d] ep %02d/%d  val_loss=%.4f  macroF1=%.4f  alert_rec=%.3f" % (
            fold, epoch + 1, EPOCHS, vloss, mf1, arec))
        if mf1 > best_f1 + 1e-4 or (abs(mf1 - best_f1) <= 1e-4 and vloss < best_vloss):
            best_f1 = mf1; best_vloss = vloss; best_state = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print("  [fold %d] early stop @ ep %d (best macroF1=%.4f)" % (fold, epoch + 1, best_f1))
                break
    return best_state, best_f1

y_pool = np.array([m["label"] for m in trainval_items])
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_f1 = []; g_best_f1 = -1.0; g_best_state = None
for fold, (tr_idx, va_idx) in enumerate(skf.split(trainval_items, y_pool), 1):
    tr_items = [trainval_items[i] for i in tr_idx]
    va_items = [trainval_items[i] for i in va_idx]
    print("Fold %d/%d  train=%d  val=%d" % (fold, N_SPLITS, len(tr_items), len(va_items)))
    state, f1 = train_one_fold_f1(tr_items, va_items, fold)
    fold_f1.append(f1)
    if f1 > g_best_f1:
        g_best_f1 = f1; g_best_state = state
        print("  ** new global best macroF1=%.4f (fold %d) **" % (f1, fold))

print("\nCV macroF1: %.4f +/- %.4f" % (float(np.mean(fold_f1)), float(np.std(fold_f1))))
torch.save({"model_state_dict": g_best_state, "class_names": CLASS_NAMES,
    "alert_class_indices": alert_idx, "val_macro_f1": float(g_best_f1),
    "cv_val_macro_f1_mean": float(np.mean(fold_f1)), "selection": "macro_f1",
    "config": {"sr": SR, "duration": DURATION, "n_fft": N_FFT, "hop": HOP,
        "n_mels": N_MELS, "normalize": "per_clip_zscore"}}, macrof1_save_path)
print("Saved macro-F1-selected model ->", macrof1_save_path)


# === CELL B  -  Evaluate the macro-F1 model on the 25% held-out test set ===
import numpy as np, torch
from torch.utils.data import DataLoader
ckpt = torch.load(macrof1_save_path, map_location=device)
model = AudioCNN(NUM_CLASSES).to(device)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
test_loader = DataLoader(MelDataset(test_items, augment=False),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
with torch.no_grad():
    for x, y in test_loader:
        p = model(x.to(device)).argmax(1).cpu().numpy()
        for t, pp in zip(y.numpy(), p): cm[t, pp] += 1
print("HELD-OUT TEST (25%)  [macro-F1 model]  n =", int(cm.sum()))
print("%-10s %6s %6s %6s %6s" % ("class", "prec", "rec", "f1", "n"))
f1s = []
for i, c in enumerate(CLASS_NAMES):
    tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9); f1s.append(f1)
    flag = "  <-- ALERT" if i in ckpt["alert_class_indices"] else ""
    print("%-10s %6.3f %6.3f %6.3f %6d%s" % (c, prec, rec, f1, cm[i, :].sum(), flag))
print("\nTest acc:", round(float(np.trace(cm)) / max(int(cm.sum()), 1), 4),
      "| test macro-F1:", round(float(np.mean(f1s)), 4),
      "| CV macro-F1 mean:", round(ckpt["cv_val_macro_f1_mean"], 4))
