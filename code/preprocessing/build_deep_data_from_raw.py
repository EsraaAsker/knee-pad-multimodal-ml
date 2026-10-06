from pathlib import Path
import argparse
import json
import numpy as np

def resample_channels(a, target):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] < 1:
        raise ValueError(a.shape)
    if a.shape[1] == target:
        return a
    old = np.linspace(0.0, 1.0, a.shape[1])
    new = np.linspace(0.0, 1.0, target)
    return np.vstack([np.interp(new, old, row) for row in a]).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imu-len", type=int, default=593)
    ap.add_argument("--emg-len", type=int, default=5037)
    ap.add_argument("--max-label", type=int, default=8)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for p in sorted((raw / "dataset").glob("Subject_*/[0-9]/Trial_*/imu.npy")):
        subject, label, trial = p.relative_to(raw / "dataset").parts[:3]
        label = int(label)
        if label > args.max_label:
            continue
        emg_path = p.with_name("emg.npy")
        if not emg_path.exists():
            continue
        im = np.load(p, allow_pickle=False)
        em = np.load(emg_path, allow_pickle=False)
        if im.shape[0] != 48 or em.shape[0] != 8:
            continue
        records.append((p, subject, label, trial))

    if not records:
        raise SystemExit("No valid records found.")

    n = len(records)
    imu_out = np.memmap(out / "imu_official.dat", dtype="float32", mode="w+", shape=(n, 48, args.imu_len))
    emg_out = np.memmap(out / "emg_official.dat", dtype="float32", mode="w+", shape=(n, 8, args.emg_len))
    labels = np.empty(n, dtype=np.int64)
    trial_ids = np.empty(n, dtype=object)
    subjects = np.empty(n, dtype=object)

    for i, (p, subject, label, trial) in enumerate(records):
        imu_out[i] = resample_channels(np.load(p, allow_pickle=False), args.imu_len)
        emg_out[i] = resample_channels(np.load(p.with_name("emg.npy"), allow_pickle=False), args.emg_len)
        labels[i] = label
        trial_ids[i] = f"{subject}/{label}/{trial}"
        subjects[i] = subject

    imu_out.flush()
    emg_out.flush()
    np.save(out / "labels.npy", labels)
    np.save(out / "trial_ids.npy", trial_ids)
    np.save(out / "subjects.npy", subjects)

    manifest = {
        "n_records": n,
        "imu_shape": [n, 48, args.imu_len],
        "emg_shape": [n, 8, args.emg_len],
        "label_encoding": "raw directory labels 0..8 retained",
        "record_granularity": "one resampled record per Subject/label/Trial",
        "resampling": "linear interpolation over normalized trial time"
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
