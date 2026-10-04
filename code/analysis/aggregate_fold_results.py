from pathlib import Path
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = Path(os.environ.get("KNEEPAD_RESULTS", ROOT / "results"))
OUT = Path(os.environ.get("KNEEPAD_OUT", ROOT / "results" / "summaries"))
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT.rglob("fold_metrics.csv"))
frames = [pd.read_csv(p) for p in files if p.stat().st_size > 0]
if frames:
    pd.concat(frames, ignore_index=True).to_csv(OUT / "all_fold_metrics.csv", index=False)
