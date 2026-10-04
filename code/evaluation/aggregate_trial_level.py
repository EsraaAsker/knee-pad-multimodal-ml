from pathlib import Path
import os
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
INPUT = Path(os.environ.get("KNEEPAD_PREDICTIONS", ROOT / "results" / "strict" / "predictions"))
OUT = Path(os.environ.get("KNEEPAD_OUT", ROOT / "results" / "diagnostics"))
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for path in sorted(INPUT.glob("*.csv")):
    d = pd.read_csv(path)
    if not {"true_label", "prob_wrong", "trial_id"}.issubset(d.columns):
        continue
    d["pred_wrong"] = (d["prob_wrong"] >= 0.5).astype(int)
    d["true_wrong"] = (d["true_label"] != 0).astype(int)
    pooled = d.groupby(["trial_id", "subject_id"], as_index=False).agg(
        true_wrong=("true_wrong", "first"),
        prob_wrong=("prob_wrong", "mean")
    )
    pooled["pred_wrong"] = (pooled["prob_wrong"] >= 0.5).astype(int)
    rows.append(pooled)

if not rows:
    raise SystemExit("No compatible prediction files found.")

z = pd.concat(rows, ignore_index=True)
z.to_csv(OUT / "trial_level_predictions.csv", index=False)
summary = pd.DataFrame([{
    "accuracy": accuracy_score(z.true_wrong, z.pred_wrong),
    "balanced_accuracy": balanced_accuracy_score(z.true_wrong, z.pred_wrong),
    "macro_f1": f1_score(z.true_wrong, z.pred_wrong, average="macro"),
    "n_trials": len(z)
}])
summary.to_csv(OUT / "trial_level_metrics.csv", index=False)
