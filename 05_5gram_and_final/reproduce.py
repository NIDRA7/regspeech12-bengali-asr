"""
reproduce.py
Fast verification of 72.3% Mean WER from saved 5-gram predictions.
No GPU, no decoding. ~5 sec.
Setup: pip install "numpy<2" jiwer pandas
"""
import pickle, re, unicodedata, os
import numpy as np, pandas as pd, jiwer

HERE = os.path.dirname(os.path.abspath(__file__))
with open(HERE + "/test_preds_5gram.pkl", "rb") as f:
    data = pickle.load(f)
preds, refs, dialects = data["preds_5gram"], data["refs"], data["dialects"]

PUNCT = '।,.!?-"\'()[]{}<>:;/\\|@#$%^&*+=~`'
PUNCT_SET = set(PUNCT)
def normalize(text):
    if pd.isna(text):
        return " "
    text = unicodedata.normalize("NFC", str(text))
    text = "".join(c if c not in PUNCT_SET else " " for c in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else " "

per = {}
for d in sorted(set(x for x in dialects if x)):
    idx = [i for i, dd in enumerate(dialects) if dd == d]
    r = [normalize(refs[i]) for i in idx]
    p = [normalize(preds[i]) for i in idx]
    per[d] = round(100 * jiwer.wer(r, p), 1)
mean = round(sum(per.values()) / len(per), 1)
for d in sorted(per, key=lambda x: per[x]):
    print(d, per[d])
print("MEAN WER:", mean, "(expected 72.3%)")
