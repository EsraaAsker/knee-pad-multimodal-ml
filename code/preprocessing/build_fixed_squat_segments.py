from pathlib import Path
import argparse
import json
import numpy as np

WINDOW = 593
STRIDE = 37

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Path containing the KneE-PAD dataset directory.")
    ap.add_argument("--out", required=True, help="Output directory for processed arrays.")
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for p in sorted((raw / "dataset").glob("Subject_*/[0-2]/Trial_*/imu.npy")):
        subject, label, trial = p.relative_to(raw / "dataset").parts[:3]
        x = np.load(p, allow_pickle=False).astype("float32")
        length = x.shape[1]
        starts = [0] if length < WINDOW else list(range(0, max(1, length - WINDOW + 1), STRIDE))
        for start in starts:
            z = x[:, start:min(start + WINDOW, length)]
            if z.shape[1] < WINDOW:
                z = np.pad(z, ((0, 0), (0, WINDOW - z.shape[1])), mode="edge")
            records.append((z, int(label), f"{subject}/{label}/{trial}", subject, start))

    n = len(records)
    imu = np.memmap(out / "imu_official.dat", dtype="float32", mode="w+", shape=(n, 48, WINDOW))
    emg = np.memmap(out / "emg_official.dat", dtype="float32", mode="w+", shape=(n, 8, 5037))
    labels = np.empty(n, dtype=np.int64)
    trial_ids = np.empty(n, dtype=object)
    subjects = np.empty(n, dtype=object)
    starts = np.empty(n, dtype=np.int64)
    emg[:] = 0

    for i, (segment, label, trial_id, subject, start) in enumerate(records):
        imu[i] = segment
        labels[i] = label
        trial_ids[i] = trial_id
        subjects[i] = subject
        starts[i] = start

    imu.flush()
    emg.flush()
    np.save(out / "labels.npy", labels)
    np.save(out / "trial_ids.npy", trial_ids)
    np.save(out / "subjects.npy", subjects)
    np.save(out / "segment_starts.npy", starts)

    manifest = {
        "n_segments": n,
        "imu_shape": [n, 48, WINDOW],
        "emg_shape": [n, 8, 5037],
        "window_samples": WINDOW,
        "stride_samples": STRIDE,
        "labels": {str(k): int((labels == k).sum()) for k in sorted(set(labels.tolist()))},
        "segment_order": "sorted Subject/label/Trial then ascending window start"
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
