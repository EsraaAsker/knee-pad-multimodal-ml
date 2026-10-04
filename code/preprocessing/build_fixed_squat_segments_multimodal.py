from pathlib import Path
import argparse
import json
import numpy as np

WINDOW = 593
STRIDE = 37
EMG_LENGTH = 5037

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for p in sorted((raw / "dataset").glob("Subject_*/[0-2]/Trial_*/imu.npy")):
        subject, label, trial = p.relative_to(raw / "dataset").parts[:3]
        im = np.load(p, allow_pickle=False).astype("float32")
        em = np.load(p.with_name("emg.npy"), allow_pickle=False).astype("float32")
        length = im.shape[1]
        starts = [0] if length < WINDOW else list(range(0, max(1, length - WINDOW + 1), STRIDE))

        for start in starts:
            z = im[:, start:min(start + WINDOW, length)]
            if z.shape[1] < WINDOW:
                z = np.pad(z, ((0, 0), (0, WINDOW - z.shape[1])), mode="edge")
            lo = start / max(length, 1)
            hi = min(1.0, (start + WINDOW) / max(length, 1))
            a = max(0, int(round(lo * (em.shape[1] - 1))))
            b = max(a + 1, int(round(hi * (em.shape[1] - 1))))
            ez = em[:, a:b]
            old = np.linspace(0, 1, ez.shape[1])
            new = np.linspace(0, 1, EMG_LENGTH)
            er = np.vstack([np.interp(new, old, row) for row in ez]).astype("float32")
            records.append((z, er, int(label), f"{subject}/{label}/{trial}", subject, start))

    n = len(records)
    imu = np.memmap(out / "imu_official.dat", dtype="float32", mode="w+", shape=(n, 48, WINDOW))
    emg = np.memmap(out / "emg_official.dat", dtype="float32", mode="w+", shape=(n, 8, EMG_LENGTH))
    labels = np.empty(n, "int64")
    trial_ids = np.empty(n, object)
    subjects = np.empty(n, object)
    starts = np.empty(n, "int64")

    for i, (im, em, label, trial_id, subject, start) in enumerate(records):
        imu[i], emg[i] = im, em
        labels[i], trial_ids[i], subjects[i], starts[i] = label, trial_id, subject, start

    imu.flush(); emg.flush()
    np.save(out / "labels.npy", labels)
    np.save(out / "trial_ids.npy", trial_ids)
    np.save(out / "subjects.npy", subjects)
    np.save(out / "segment_starts.npy", starts)
    (out / "manifest.json").write_text(json.dumps({
        "n_segments": n,
        "imu_shape": [n, 48, WINDOW],
        "emg_shape": [n, 8, EMG_LENGTH],
        "window_samples": WINDOW,
        "stride_samples": STRIDE,
        "emg_alignment": "raw EMG interval matched to the normalized IMU window and resampled to 5037 points"
    }, indent=2), encoding="utf-8")
    print(n)

if __name__ == "__main__":
    main()
