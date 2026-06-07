"""
reproduce.py
Verifies the 82.5% greedy / 78.8% LM Mean WER from saved predictions CSV.
No GPU needed. ~10 seconds.

Usage:
    pip install jiwer pandas
    python reproduce.py
"""
import pandas as pd, jiwer, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f"{HERE}/test_predictions_full.csv")
df["reference"] = df["reference"].fillna("").astype(str)
df["greedy_prediction"] = df["greedy_prediction"].fillna(" ").astype(str)
df["lm_prediction"] = df["lm_prediction"].fillna(" ").astype(str)

per_dialect = {}
for d in sorted(df["dialect"].dropna().unique()):
    sub = df[df["dialect"] == d]
    refs = sub["reference"].tolist()
    g = [p if p.strip() else " " for p in sub["greedy_prediction"].tolist()]
    l = [p if p.strip() else " " for p in sub["lm_prediction"].tolist()]
    per_dialect[d] = {"n": len(refs),
                      "greedy": round(100 * jiwer.wer(refs, g), 1),
                      "lm":     round(100 * jiwer.wer(refs, l), 1)}

mean_g = round(np.mean([v["greedy"] for v in per_dialect.values()]), 1)
mean_l = round(np.mean([v["lm"]     for v in per_dialect.values()]), 1)

print(f"{'Dialect':<14} {'N':>5} {'Greedy':>10} {'+KenLM':>10}")
print("-" * 45)
for d, v in sorted(per_dialect.items(), key=lambda x: x[1]["greedy"]):
    print(f"{d:<14} {v['n']:>5} {v['greedy']:>9.1f}% {v['lm']:>9.1f}%")
print("-" * 45)
print(f"{'MEAN WER':<14} {'':>5} {mean_g:>9.1f}% {mean_l:>9.1f}%")
print("(Expected: 82.5% / 78.8%)")
