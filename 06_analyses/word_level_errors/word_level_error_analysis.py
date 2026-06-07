"""
word_level_error_analysis.py
Word-level error breakdown (Substitution/Insertion/Deletion) per dialect
+ Bengali-specific character error patterns (conjuncts, nukta, vowels)
+ correlation of deletions/substitutions with WER.

CPU only, ~1 min. Setup:
    pip install "numpy<2" jiwer pandas scipy
Needs: test_preds_5gram.pkl (in this folder)
"""
import os, pickle, re, unicodedata, json
import numpy as np, pandas as pd, jiwer
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
with open(HERE + "/test_preds_5gram.pkl", "rb") as f:
    best = pickle.load(f)
preds, refs, dialects = best["preds_5gram"], best["refs"], best["dialects"]

PUNCT = set('।,.!?-"\'()[]{}<>:;/\\|@#$%^&*+=~`')
def norm(text):
    if pd.isna(text):
        return " "
    text = unicodedata.normalize("NFC", str(text))
    text = "".join(c if c not in PUNCT else " " for c in text)
    return re.sub(r"\s+", " ", text).strip() or " "

err_rows = []
for d in sorted(set(x for x in dialects if x)):
    idx = [i for i, dd in enumerate(dialects) if dd == d]
    r = [norm(refs[i]) for i in idx]; p = [norm(preds[i]) for i in idx]
    out = jiwer.process_words(r, p)
    nref = out.substitutions + out.deletions + out.hits
    err_rows.append({"dialect": d, "N": len(idx),
                     "sub_pct": round(100*out.substitutions/nref,1),
                     "ins_pct": round(100*out.insertions/nref,1),
                     "del_pct": round(100*out.deletions/nref,1),
                     "wer": round(100*out.wer,1)})
edf = pd.DataFrame(err_rows)

VOWEL_SIGNS = set("ািীুূৃেৈোৌ")
NUKTA = "়"; HASANTA = "্"
INDEP = set("অআইঈউঊঋএঐওঔ")
def char_classes(text):
    c = {"vowel_signs":0,"nukta":0,"conjuncts":0,"indep_vowels":0,"consonants":0}
    for ch in text:
        if ch in VOWEL_SIGNS: c["vowel_signs"] += 1
        elif ch == NUKTA: c["nukta"] += 1
        elif ch == HASANTA: c["conjuncts"] += 1
        elif ch in INDEP: c["indep_vowels"] += 1
        elif "ক" <= ch <= "হ": c["consonants"] += 1
    return c
ref_all = " ".join(norm(refs[i]) for i in range(len(refs)))
pred_all = " ".join(norm(preds[i]) for i in range(len(preds)))
rc, pc = char_classes(ref_all), char_classes(pred_all)

r_del, p_del = pearsonr(edf["del_pct"], edf["wer"])
r_sub, p_sub = pearsonr(edf["sub_pct"], edf["wer"])

print("WORD-LEVEL ERROR BREAKDOWN")
print(edf.sort_values("wer").to_string(index=False))
print("\nDeletions vs WER:  r=%.3f p=%.4f" % (r_del, p_del))
print("Substit. vs WER:   r=%.3f p=%.4f" % (r_sub, p_sub))
print("\nBENGALI CHARACTER RATIOS (Pred/Ref):")
for k in ["consonants","vowel_signs","conjuncts","nukta","indep_vowels"]:
    print("  %-14s %.3f" % (k, pc[k]/rc[k] if rc[k] else 0))
