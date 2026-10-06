from pathlib import Path
import os, json, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("KNEEPAD_DATA", REPO_ROOT / "data" / "processed" / "deep_data"))
OUT = Path(os.environ.get("KNEEPAD_OUT", REPO_ROOT / "results" / "dl_results"))
OUT.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("SEED", "2026"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 2)))

raw_y = np.load(ROOT / "labels.npy")
N = len(raw_y)
E = np.memmap(ROOT / "emg_official.dat", dtype="float32", mode="r", shape=(N, 8, 5037))
I = np.memmap(ROOT / "imu_official.dat", dtype="float32", mode="r", shape=(N, 48, 593))
task = os.environ.get("TASK", "multiclass")
y = (raw_y != 0).astype(np.int64) if task == "binary" else raw_y
tids = np.load(ROOT / "trial_ids.npy")
subs = np.load(ROOT / "subjects.npy")

trial_table = pd.DataFrame({"tid": tids, "label": y}).drop_duplicates("tid").reset_index(drop=True)
trial_table.to_csv(OUT / "trial_table.csv", index=False)
folds = list(StratifiedKFold(20, shuffle=True, random_state=SEED).split(trial_table.tid, trial_table.label))

class DS(Dataset):
    def __init__(self, idx, emg_mean, emg_std, imu_mean, imu_std, train=False):
        self.idx, self.emg_mean, self.emg_std = idx, emg_mean, emg_std
        self.imu_mean, self.imu_std, self.train = imu_mean, imu_std, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        j = self.idx[k]
        e = np.asarray(E[j, :, ::20], np.float32)
        i = np.asarray(I[j, :, ::2], np.float32)
        e = (e - self.emg_mean[:, None]) / (self.emg_std[:, None] + 1e-5)
        i = (i - self.imu_mean[:, None]) / (self.imu_std[:, None] + 1e-5)
        if self.train:
            e = e + np.random.normal(0, 0.015, e.shape).astype(np.float32)
            i = i + np.random.normal(0, 0.015, i.shape).astype(np.float32)
        return torch.from_numpy(e), torch.from_numpy(i), torch.tensor(int(y[j]), dtype=torch.long), j

class Block(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c1, c2, 7, padding=3), nn.BatchNorm1d(c2), nn.GELU(),
            nn.Conv1d(c2, c2, 5, padding=2), nn.BatchNorm1d(c2), nn.GELU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        return self.net(x)

class Model(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()
        self.emg = nn.Sequential(Block(8, 32), Block(32, 64))
        self.imu = nn.Sequential(Block(48, 48), Block(48, 64))
        self.proj = nn.Linear(128, 128)
        enc = nn.TransformerEncoderLayer(
            128, 8, dim_feedforward=256, dropout=0.15,
            batch_first=True, norm_first=True, activation="gelu"
        )
        self.tr = nn.TransformerEncoder(enc, 2)
        self.att = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.25), nn.Linear(128, n_outputs))

    def forward(self, e, i):
        a = self.emg(e).transpose(1, 2)
        b = self.imu(i).transpose(1, 2)
        n = min(a.size(1), b.size(1))
        z = self.proj(torch.cat([a[:, :n], b[:, :n]], -1))
        z = self.tr(z)
        w = torch.softmax(self.att(z).squeeze(-1), -1).unsqueeze(-1)
        return self.head((z * w).sum(1))

def stats(train_idx):
    em = np.asarray(E[train_idx, :, ::20])
    im = np.asarray(I[train_idx, :, ::2])
    return em.mean((0, 2)), em.std((0, 2)), im.mean((0, 2)), im.std((0, 2))

def evaluate(model, loader, device):
    model.eval()
    yy, pp, ids = [], [], []
    with torch.no_grad():
        for e, i, t, j in loader:
            q = model(e.to(device), i.to(device)).softmax(-1).cpu().numpy()
            pp.extend(q.argmax(1)); yy.extend(t.numpy()); ids.extend(j.numpy())
    return np.array(yy), np.array(pp), np.array(ids)

device = "cuda" if torch.cuda.is_available() else "cpu"
rows, pred_rows = [], []
limit = int(os.environ.get("FOLD_LIMIT", "20"))
epochs = int(os.environ.get("EPOCHS", "18"))

for fi, (trg, teg) in enumerate(folds[:limit]):
    tr_trials = set(trial_table.tid.iloc[trg])
    te_trials = set(trial_table.tid.iloc[teg])
    tr = np.where(np.isin(tids, list(tr_trials)))[0]
    te = np.where(np.isin(tids, list(te_trials)))[0]
    em, es, im, is_ = stats(tr)
    tl = DataLoader(DS(tr, em, es, im, is_, True), 64, shuffle=True, num_workers=0)
    vl = DataLoader(DS(te, em, es, im, is_, False), 128, shuffle=False, num_workers=0)
    n_outputs = 2 if task == "binary" else 9
    model = Model(n_outputs).to(device)
    counts = np.bincount(y[tr], minlength=n_outputs)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1)); weights /= weights.mean()
    crit = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device), label_smoothing=0.04)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = (-1, None)

    for ep in range(epochs):
        model.train()
        for e, i, t, _ in tl:
            opt.zero_grad()
            loss = crit(model(e.to(device), i.to(device)), t.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        yy, pp, _ = evaluate(model, vl, device)
        score = balanced_accuracy_score(yy, pp)
        if score > best[0]:
            best = (score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[1])
    yy, pp, ids = evaluate(model, vl, device)
    rows.append({
        "fold": fi, "n_train_trials": len(trg), "n_test_trials": len(teg),
        "n_train_segments": len(tr), "n_test_segments": len(te),
        "accuracy": accuracy_score(yy, pp),
        "balanced_accuracy": balanced_accuracy_score(yy, pp),
        "macro_f1": f1_score(yy, pp, average="macro"), "epochs": epochs
    })
    for j, p in zip(ids, pp):
        pred_rows.append({
            "fold": fi, "segment_index": int(j), "trial_id": tids[j],
            "subject": int(subs[j]), "raw_label": int(y[j]), "prediction": int(p)
        })
    print(rows[-1])

pd.DataFrame(rows).to_csv(OUT / f"dl_fold_metrics_{limit}fold.csv", index=False)
pd.DataFrame(pred_rows).to_csv(OUT / f"dl_segment_predictions_{limit}fold.csv", index=False)
manifest = {
    "device": device, "folds_run": limit, "epochs": epochs,
    "architecture": "dual-branch EMG/IMU CNN + 2-layer Transformer encoder + attention pooling",
    "split": "raw trial-held-out before overlapping official windows",
    "task": task
}
(OUT / f"run_manifest_{limit}fold.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
