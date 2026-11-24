"""Reproduce the Stage 1: Zero-shot IndicWav2Vec row of Table 3 (Raw and BnUnicode: WER, CER, SD).
Usage: python reproduce.py"""
import sys, os, glob, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from scoring import score_by_dialect, check
HERE = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
df = pd.read_csv(os.path.join(HERE, "predictions", "zero_shot_predictions.csv"), keep_default_na=False)
refs, preds, dia = df["reference"].tolist(), df["prediction"].tolist(), df["dialect"].tolist()
ok = True
for name, norm, exp in (("raw", "raw", (89.0, 60.1, 11.3)), ("bnunicode", "bnunicode", (86.6, 58.7, 13.3))):
    r = score_by_dialect(refs, preds, dia, norm)
    ok &= check("Stage 1 " + name, (r["macro_wer"], r["macro_cer"], r["sd_wer"]), exp)
sys.exit(0 if ok else 1)
