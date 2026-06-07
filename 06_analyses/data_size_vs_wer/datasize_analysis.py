"""
datasize_analysis.py
Tests whether per-dialect training data size predicts WER (best result).
Finding: it does NOT - linguistic divergence matters more than data size.

Setup: pip install "numpy<2" pandas scipy openpyxl
Needs: dataset/train.xlsx (for per-dialect training counts)
"""
import re
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

best_wer = {"comilla":43.4,"tangail":45.3,"habiganj":65.0,"narail":70.3,"narsingdi":71.5,
            "barishal":74.4,"sylhet":74.9,"rangpur":80.8,"noakhali":82.2,"sandwip":85.0,
            "kishoreganj":87.3,"chittagong":87.4}
def gd(fn):
    m = re.search(r"train_([a-z]+)_", str(fn)); return m.group(1) if m else None
train_df = pd.read_excel("train.xlsx")
train_df["dialect"] = train_df["file_name"].apply(gd)
tc = train_df["dialect"].value_counts().to_dict()
rows = [{"dialect":d,"train_samples":tc.get(d,0),"wer":best_wer[d]} for d in best_wer]
df = pd.DataFrame(rows).sort_values("train_samples")
ta = df["train_samples"].values.astype(float); wa = df["wer"].values
print(df.to_string(index=False))
print("Pearson  r=%.3f p=%.4f" % pearsonr(ta,wa))
print("Spearman r=%.3f p=%.4f" % spearmanr(ta,wa))
print("Pearson(log) r=%.3f p=%.4f" % pearsonr(np.log(ta),wa))
