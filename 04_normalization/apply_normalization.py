"""
apply_normalization.py
============================================================
Applies safe text normalization (NFC + remove punctuation +
collapse whitespace) to predictions + references, then recomputes
Mean WER for all 4 stages of the pipeline.

PRODUCES:
  Zero-shot baseline:  89.0% -> 86.8%
  Fine-tune (greedy):  82.5% -> 77.7%
  + Global KenLM:      78.8% -> 72.9%
  + Dialect-aware LM:  78.6% -> 73.1%   <-- HEADLINE

Setup: pip install jiwer pandas numpy
============================================================
"""
import pandas as pd, jiwer, re, unicodedata, os
import numpy as np

MAIN_DIR = "/content/drive/MyDrive/claudefinetuneandklmn"

def normalize(text):
    if pd.isna(text):
        return " "
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"[।,.!?\-\"\'\(\)\[\]\{\}<>:;/\\|@#$%^&*+=~`]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else " "

def get_dialect(fn):
    m = re.search(r"test_([a-z]+)_", str(fn))
    return m.group(1) if m else None

def macro_wer(df, ref_col, pred_col, apply_norm=False):
    if "dialect" not in df.columns:
        df["dialect"] = df["file_name"].apply(get_dialect)
    per = {}
    for d in sorted(df["dialect"].dropna().unique()):
        sub = df[df["dialect"] == d]
        refs = sub[ref_col].tolist()
        preds = sub[pred_col].tolist()
        if apply_norm:
            refs = [normalize(r) for r in refs]
            preds = [normalize(p) for p in preds]
        else:
            refs = [str(r) if pd.notna(r) else " " for r in refs]
            preds = [str(p) if pd.notna(p) and str(p).strip() else " " for p in preds]
        per[d] = round(100 * jiwer.wer(refs, preds), 1)
    return round(np.mean(list(per.values())), 1), per

df_b = pd.read_csv(f"{MAIN_DIR}/baselineperformance/baseline_predictions_full.csv")
df_lm = pd.read_csv(f"{MAIN_DIR}/test_predictions.csv")
df_d = pd.read_csv(f"{MAIN_DIR}/testdataglobalanddialectklmna/test_predictions_full.csv")
winner_col = next((c for c in ["winner_prediction", "winner", "dialect_prediction", "best_prediction", "prediction"] if c in df_d.columns), None)

base_old, _ = macro_wer(df_b, "reference", "prediction", False)
base_new, _ = macro_wer(df_b, "reference", "prediction", True)
g_old, _ = macro_wer(df_lm, "reference", "greedy_prediction", False)
g_new, _ = macro_wer(df_lm, "reference", "greedy_prediction", True)
gl_old, _ = macro_wer(df_lm, "reference", "lm_prediction", False)
gl_new, _ = macro_wer(df_lm, "reference", "lm_prediction", True)
d_old, d_per_old = macro_wer(df_d, "reference", winner_col, False)
d_new, d_per_new = macro_wer(df_d, "reference", winner_col, True)

print("=" * 60)
print("Mean WER - Before vs After Normalization")
print("=" * 60)
print(f"Zero-shot baseline:   {base_old:.1f}% -> {base_new:.1f}%")
print(f"Fine-tune (greedy):   {g_old:.1f}% -> {g_new:.1f}%")
print(f"+ Global KenLM:       {gl_old:.1f}% -> {gl_new:.1f}%")
print(f"+ Dialect-aware LM:   {d_old:.1f}% -> {d_new:.1f}%")
