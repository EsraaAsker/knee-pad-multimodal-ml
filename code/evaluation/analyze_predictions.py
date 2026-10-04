from pathlib import Path
import os
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
INPUT = Path(os.environ.get("KNEEPAD_PREDICTIONS", ROOT / "results" / "strict" / "predictions"))
OUT = Path(os.environ.get("KNEEPAD_OUT", ROOT / "results" / "diagnostics"))
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT.glob("*.csv"))
if not files:
    raise SystemExit(f"No prediction CSV files found in {INPUT}")

allp = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
y = allp["true_label"].to_numpy()
p = allp["predicted_label"].to_numpy()
wrong = (y != 0).astype(int)
pred_wrong = (p != 0).astype(int)

metrics = pd.DataFrame([{
    "accuracy": accuracy_score(y, p),
    "balanced_accuracy": balanced_accuracy_score(y, p),
    "macro_f1": f1_score(y, p, average="macro"),
    "correct_recall": ((y == 0) & (p == 0)).sum() / max(1, (y == 0).sum()),
    "wrong_recall": ((y != 0) & (p != 0)).sum() / max(1, (y != 0).sum())
}])
metrics.to_csv(OUT / "prediction_metrics.csv", index=False)

cm = confusion_matrix(wrong, pred_wrong, labels=[0, 1])
pd.DataFrame(cm, index=["Correct", "Wrong"], columns=["Pred Correct", "Pred Wrong"]).to_csv(
    OUT / "binary_confusion_matrix.csv"
)
