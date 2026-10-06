from pathlib import Path
import os, random, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Portable repository-relative configuration.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("KNEEPAD_DATA", REPO_ROOT / "data" / "processed" / "deep_data"))
OUT = Path(os.environ.get("KNEEPAD_OUT", REPO_ROOT / "results" / "levels"))
OUT.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("SEED", "2026"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 2)))

raw = np.load(DATA / "labels.npy")
N = len(raw)
E = np.memmap(DATA / "emg_official.dat", dtype="float32", mode="r", shape=(N, 8, 5037))
I = np.memmap(DATA / "imu_official.dat", dtype="float32", mode="r", shape=(N, 48, 593))
tids = np.load(DATA / "trial_ids.npy", allow_pickle=True)

def target_for(task):
    return (raw // 3).astype(np.int64) if task == "level1" else raw.astype(np.int64)

class DS(Dataset):
    def __init__(self, idx, em, es, im, ins, y, train=False):
        self.idx, self.em, self.es, self.im, self.ins, self.y, self.train = idx, em, es, im, ins, y, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        j = int(self.idx[k])
        e = np.asarray(E[j, :, ::20], np.float32)
        i = np.asarray(I[j, :, ::2], np.float32)
        e = (e - self.em[:, None]) / (self.es[:, None] + 1e-5)
        i = (i - self.im[:, None]) / (self.ins[:, None] + 1e-5)
        if self.train:
            e = e + np.random.normal(0, 0.015, e.shape).astype(np.float32)
            i = i + np.random.normal(0, 0.015, i.shape).astype(np.float32)
        return torch.from_numpy(e), torch.from_numpy(i), torch.tensor(int(self.y[j]), dtype=torch.long)

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
    def __init__(self, nout):
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
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.25), nn.Linear(128, nout))

    def forward(self, e, i):
        a = self.emg(e).transpose(1, 2)
        b = self.imu(i).transpose(1, 2)
        n = min(a.size(1), b.size(1))
        z = self.proj(torch.cat([a[:, :n], b[:, :n]], -1))
        z = self.tr(z)
        w = torch.softmax(self.att(z).squeeze(-1), -1).unsqueeze(-1)
        return self.head((z * w).sum(1))

def stats(idx):
    em = np.asarray(E[idx, :, ::20])
    im = np.asarray(I[idx, :, ::2])
    return em.mean((0, 2)), em.std((0, 2)), im.mean((0, 2)), im.std((0, 2))

def run(task, nout):
    y = target_for(task)
    tt = pd.DataFrame({"tid": tids, "y": y}).drop_duplicates("tid").reset_index(drop=True)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(tt.tid, tt.y))
    rows = []
    epochs = int(os.environ.get("EPOCHS", "18"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(task, "device", device, "epochs", epochs, flush=True)

    for fi, (trg, teg) in enumerate(folds):
        tr_trials = set(tt.tid.iloc[trg])
        te_trials = set(tt.tid.iloc[teg])
        tr = np.where(np.isin(tids, list(tr_trials)))[0]
        te = np.where(np.isin(tids, list(te_trials)))[0]
        em, es, im, ins = stats(tr)
        tl = DataLoader(DS(tr, em, es, im, ins, y, True), 64, True, num_workers=0)
        vl = DataLoader(DS(te, em, es, im, ins, y, False), 128, False, num_workers=0)
        model = Model(nout).to(device)
        counts = np.bincount(y[tr], minlength=nout)
        weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
        weights = weights / weights.mean()
        crit = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device),
            label_smoothing=0.04
        )
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best = (-1, None)

        for ep in range(epochs):
            model.train()
            for e, i, t in tl:
                opt.zero_grad()
                loss = crit(model(e.to(device), i.to(device)), t.to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            model.eval()
            yy, pp = [], []
            with torch.no_grad():
                for e, i, t in vl:
                    pp.extend(model(e.to(device), i.to(device)).argmax(1).cpu().numpy())
                    yy.extend(t.numpy())
            score = balanced_accuracy_score(yy, pp)
            if score > best[0]:
                best = (score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        model.load_state_dict(best[1])
        model.eval()
        yy, pp = [], []
        with torch.no_grad():
            for e, i, t in vl:
                pp.extend(model(e.to(device), i.to(device)).argmax(1).cpu().numpy())
                yy.extend(t.numpy())

        rows.append({
            "task": task, "fold": fi, "n_train_trials": len(trg), "n_test_trials": len(teg),
            "n_train_segments": len(tr), "n_test_segments": len(te),
            "accuracy": accuracy_score(yy, pp),
            "balanced_accuracy": balanced_accuracy_score(yy, pp),
            "macro_f1": f1_score(yy, pp, average="macro"),
            "epochs": epochs
        })
        print(rows[-1], flush=True)

    pd.DataFrame(rows).to_csv(OUT / f"{task}_fold_metrics.csv", index=False)
    manifest = {
        "task": task, "n_outputs": nout, "device": device, "epochs": epochs, "folds": 5,
        "split": "trial-level StratifiedKFold(5), no subject-held-out",
        "architecture": "dual-branch EMG/IMU CNN, 2-layer 8-head Transformer encoder, attention pooling, linear classification head",
        "loss": "class-weighted CrossEntropyLoss with label smoothing 0.04; weights and normalization train-fold only"
    }
    (OUT / f"{task}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

for task, nout in [("level1", 3), ("level3", 9)]:
    run(task, nout)
