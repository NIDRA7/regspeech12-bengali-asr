"""
reproduce.py
Reproduces the 89.0% Mean WER from saved predictions.
Takes ~5 seconds. No GPU needed.

Usage:
    pip install jiwer pandas
    python reproduce.py
"""
import pandas as pd, jiwer, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f"{HERE}/baseline_predictions_full.csv")
df["reference"] = df["reference"].fillna("").astype(str)
df["prediction"] = df["prediction"].fillna(" ").astype(str)

per_dialect = {}
for d in sorted(df["dialect"].dropna().unique()):
    sub = df[df["dialect"] == d]
    refs = sub["reference"].tolist()
    preds = [p if p.strip() else " " for p in sub["prediction"].tolist()]
    per_dialect[d] = {"n": len(refs), "wer": round(100 * jiwer.wer(refs, preds), 1)}

mean_wer = float(np.mean([v["wer"] for v in per_dialect.values()]))

print(f"{'Dialect':<14} {'N':>5} {'WER':>9}")
print("-" * 32)
for d, v in sorted(per_dialect.items(), key=lambda x: x[1]["wer"]):
    print(f"{d:<14} {v['n']:>5} {v['wer']:>8.1f}%")
print("-" * 32)
print(f"{'Mean WER':<14} {'':>5} {mean_wer:>8.1f}%  (expected: 89.0%)")
