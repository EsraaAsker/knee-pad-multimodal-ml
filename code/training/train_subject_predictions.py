from pathlib import Path
import os, json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("KNEEPAD_DATA", REPO_ROOT / "data" / "processed" / "deep_data"))
OUT = Path(os.environ.get("KNEEPAD_OUT", REPO_ROOT / "results" / "strict"))

SEED = int(os.environ.get("SEED", "2026"))
EXERCISE = int(os.environ.get("EXERCISE", "0"))
USE_EMG = int(os.environ.get("USE_EMG", "1"))
SUBSET = json.loads(os.environ.get("SUBSET", "[1,2,3,4,5,6,7,8]"))
CONFIG = os.environ.get("CONFIG", "strict")
EPOCHS = int(os.environ.get("EPOCHS", "5"))
INNER_FOLDS = int(os.environ.get("INNER_FOLDS", "1"))
THREADS = int(os.environ.get("TORCH_THREADS", str(max(1, min(8, os.cpu_count() or 2))))

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(THREADS)

raw = np.load(DATA / "labels.npy")
N = len(raw)
E = np.memmap(DATA / "emg_official.dat", dtype="float32", mode="r", shape=(N, 8, 5037))
I = np.memmap(DATA / "imu_official.dat", dtype="float32", mode="r", shape=(N, 48, 593))
trial_ids = np.load(DATA / "trial_ids.npy", allow_pickle=True)
subjects = np.load(DATA / "subjects.npy", allow_pickle=True)

mask = raw // 3 == EXERCISE
idx_all = np.where(mask)[0]
y = ((raw[idx_all] % 3) != 0).astype(np.int64)
trial_ids_sub = trial_ids[idx_all]
subject_sub = subjects[idx_all]
trial_table = pd.DataFrame({"tid": trial_ids_sub, "y": y, "sid": subject_sub}).drop_duplicates("tid").reset_index(drop=True)

outer_folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(
    trial_table.tid, trial_table.y, groups=trial_table.sid
))
imu_channels = np.array([c for s in SUBSET for c in range((s - 1) * 6, s * 6)], dtype=int)

class DS(Dataset):
    def __init__(self, idx, em, es, im, ins, train=False):
        self.idx, self.em, self.es, self.im, self.ins, self.train = np.asarray(idx), em, es, im, ins, train
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        j = int(self.idx[k])
        pos = int(np.where(idx_all == j)[0][0])
        imu = np.asarray(I[j, imu_channels, ::2], np.float32)
        imu = (imu - self.im[:, None]) / (self.ins[:, None] + 1e-5)
        if USE_EMG:
            emg = np.asarray(E[j, :, ::20], np.float32)
            emg = (emg - self.em[:, None]) / (self.es[:, None] + 1e-5)
        else:
            emg = np.zeros((1, 1), np.float32)
        if self.train:
            imu += np.random.normal(0, .015, imu.shape).astype(np.float32)
            if USE_EMG:
                emg += np.random.normal(0, .015, emg.shape).astype(np.float32)
        return torch.from_numpy(emg), torch.from_numpy(imu), torch.tensor(int(y[pos]), dtype=torch.long)

class Block(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c1, c2, 7, padding=3), nn.BatchNorm1d(c2), nn.GELU(),
            nn.Conv1d(c2, c2, 5, padding=2), nn.BatchNorm1d(c2), nn.GELU(), nn.MaxPool1d(2))
    def forward(self, x): return self.net(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_emg = USE_EMG
        self.imu = nn.Sequential(Block(len(imu_channels), 48), Block(48, 64))
        if USE_EMG:
            self.emg = nn.Sequential(Block(8, 32), Block(32, 64)); self.proj = nn.Linear(128, 128)
        else:
            self.proj = nn.Linear(64, 128)
        enc = nn.TransformerEncoderLayer(128, 8, dim_feedforward=256, dropout=.15,
                                         batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, 2)
        self.att = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Dropout(.25), nn.Linear(128, 2))
    def forward(self, e, i):
        b = self.imu(i).transpose(1, 2)
        if self.use_emg:
            a = self.emg(e).transpose(1, 2); n = min(a.size(1), b.size(1))
            z = self.proj(torch.cat([a[:, :n], b[:, :n]], -1))
        else:
            z = self.proj(b)
        z = self.tr(z)
        w = torch.softmax(self.att(z).squeeze(-1), -1).unsqueeze(-1)
        return self.head((z * w).sum(1))

def stats(train):
    imu = np.asarray(I[train][:, imu_channels, ::2], np.float32)
    imu_mean, imu_std = imu.mean((0, 2)), imu.std((0, 2))
    if USE_EMG:
        emg = np.asarray(E[train, :, ::20], np.float32)
        return emg.mean((0, 2)), emg.std((0, 2)), imu_mean, imu_std
    return np.zeros(8), np.ones(8), imu_mean, imu_std

def evaluate(model, loader, device):
    model.eval(); yy=[]; pp=[]; probs=[]
    with torch.no_grad():
        for e, i, t in loader:
            p = model(e.to(device), i.to(device)).softmax(1).cpu().numpy()
            probs.append(p); pp.extend(p.argmax(1)); yy.extend(t.numpy())
    yy, pp, probs = np.asarray(yy), np.asarray(pp), np.concatenate(probs)
    return {"accuracy": accuracy_score(yy, pp), "balanced_accuracy": balanced_accuracy_score(yy, pp),
            "macro_f1": f1_score(yy, pp, average="macro"), "y_true": yy, "y_pred": pp,
            "prob_correct": probs[:, 0], "prob_wrong": probs[:, 1]}

def criterion(train_pos, device):
    counts = np.bincount(y[train_pos], minlength=2)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1)); weights /= weights.mean()
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device), label_smoothing=.04)

def train_epochs(train_loader, val_loader, train_pos, device, epochs):
    model = Model().to(device); crit = criterion(train_pos, device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_score, best_epoch, best_state = -1, epochs, None
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        for e, i, t in train_loader:
            opt.zero_grad(); loss = crit(model(e.to(device), i.to(device)), t.to(device))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step(); met = evaluate(model, val_loader, device); history.append(met)
        if met["balanced_accuracy"] > best_score:
            best_score, best_epoch = met["balanced_accuracy"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_epoch, best_score, best_state, history

def train_fixed(train_loader, train_pos, device, epochs):
    model = Model().to(device); crit = criterion(train_pos, device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        model.train()
        for e, i, t in train_loader:
            opt.zero_grad(); loss = crit(model(e.to(device), i.to(device)), t.to(device))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
config_dir = OUT / CONFIG; config_dir.mkdir(parents=True, exist_ok=True)
rows=[]; inner_rows=[]

for fold, (tr_idx, te_idx) in enumerate(outer_folds):
    tr_trials = trial_table.iloc[tr_idx].reset_index(drop=True)
    te_trials = set(trial_table.tid.iloc[te_idx])
    splitter = GroupShuffleSplit(1, test_size=.2, random_state=SEED + fold) if INNER_FOLDS == 1 else StratifiedGroupKFold(INNER_FOLDS, shuffle=True, random_state=SEED + fold)
    scores = np.zeros(EPOCHS); counts = np.zeros(EPOCHS, dtype=int)

    for inner_fold, (itr, iva) in enumerate(splitter.split(tr_trials.tid, tr_trials.y, groups=tr_trials.sid)):
        itr_trials = set(tr_trials.tid.iloc[itr]); iva_trials = set(tr_trials.tid.iloc[iva])
        tr_pos = np.where(np.isin(trial_ids_sub, list(itr_trials)))[0]
        va_pos = np.where(np.isin(trial_ids_sub, list(iva_trials)))[0]
        gtr, gva = idx_all[tr_pos], idx_all[va_pos]
        em, es, im, ins = stats(gtr)
        tl = DataLoader(DS(gtr, em, es, im, ins, True), 64, True, num_workers=0)
        vl = DataLoader(DS(gva, em, es, im, ins), 128, False, num_workers=0)
        ep, score, _, history = train_epochs(tl, vl, tr_pos, device, EPOCHS)
        for j, metrics in enumerate(history):
            scores[j] += metrics["balanced_accuracy"]
            counts[j] += 1
        inner_rows.append({"outer_fold": fold, "inner_fold": inner_fold, "selected_epoch_inner_fold": ep, "best_inner_ba": score})

    selected_epoch = int(np.argmax(scores / np.maximum(counts, 1)) + 1)
    tr_trial_set = set(tr_trials.tid)
    tr_pos = np.where(np.isin(trial_ids_sub, list(tr_trial_set)))[0]
    gtr = idx_all[tr_pos]
    gte = idx_all[np.where(np.isin(trial_ids_sub, list(te_trials)))[0]]
    em, es, im, ins = stats(gtr)
    tl = DataLoader(DS(gtr, em, es, im, ins, True), 64, True, num_workers=0)
    tel = DataLoader(DS(gte, em, es, im, ins), 128, False, num_workers=0)
    model = train_fixed(tl, tr_pos, device, selected_epoch)
    met = evaluate(model, tel, device)

    pred_dir = OUT / "predictions"; pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "exercise": EXERCISE, "config": CONFIG, "outer_fold": fold,
        "subject_id": subjects[gte], "trial_id": trial_ids[gte],
        "true_label": met["y_true"], "predicted_label": met["y_pred"],
        "prob_correct": met["prob_correct"], "prob_wrong": met["prob_wrong"]
    }).to_csv(pred_dir / f"{CONFIG}_fold{fold}.csv", index=False)

    rows.append({"config": CONFIG, "exercise": EXERCISE, "use_emg": USE_EMG,
                 "sensor_count": len(SUBSET), "sensors": "-".join(map(str, SUBSET)),
                 "outer_fold": fold, "selected_epoch": selected_epoch,
                 "accuracy": met["accuracy"], "balanced_accuracy": met["balanced_accuracy"],
                 "macro_f1": met["macro_f1"], "epochs_max": EPOCHS, "seed": SEED,
                 "inner_folds": INNER_FOLDS, "protocol": "subject-held-out StratifiedGroupKFold"})
    pd.DataFrame(rows).to_csv(config_dir / "fold_metrics_partial.csv", index=False)

pd.DataFrame(rows).to_csv(config_dir / "fold_metrics.csv", index=False)
pd.DataFrame(inner_rows).to_csv(config_dir / "inner_epoch_selection.csv", index=False)
(config_dir / "manifest.json").write_text(json.dumps({
    "config": CONFIG, "exercise": EXERCISE, "use_emg": USE_EMG, "sensor_ids": SUBSET,
    "outer_folds": 5, "inner_folds": INNER_FOLDS, "seed": SEED, "epochs_max": EPOCHS,
    "split": "trial-level nested CV with participant-grouped outer folds",
    "normalization": "training partition only",
    "selection_metric": "inner balanced accuracy"
}, indent=2), encoding="utf-8")
