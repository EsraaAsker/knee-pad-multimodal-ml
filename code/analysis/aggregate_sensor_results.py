from pathlib import Path
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = Path(os.environ.get("KNEEPAD_RESULTS", ROOT / "results" / "strict"))
OUT = Path(os.environ.get("KNEEPAD_OUT", ROOT / "results" / "summaries"))
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT.rglob("fold_metrics.csv"))
frames = [pd.read_csv(p) for p in files if p.stat().st_size > 0]
if not frames:
    raise SystemExit("No sensor-configuration fold metrics found.")
z = pd.concat(frames, ignore_index=True)
z.groupby(["config", "sensor_count"], as_index=False).agg(
    accuracy_mean=("accuracy", "mean"),
    accuracy_sd=("accuracy", "std"),
    balanced_accuracy_mean=("balanced_accuracy", "mean"),
    balanced_accuracy_sd=("balanced_accuracy", "std"),
    macro_f1_mean=("macro_f1", "mean"),
    macro_f1_sd=("macro_f1", "std"),
    n_outer_folds=("outer_fold", "count"),
    n_seeds=("seed", "nunique")
).to_csv(OUT / "sensor_configuration_summary.csv", index=False)
