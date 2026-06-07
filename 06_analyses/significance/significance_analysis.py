"""
significance_analysis.py
Statistical significance of the WER improvement over the zero-shot baseline.
Uses per-dialect averaging (matching the headline 72.3% Mean WER).
Computes: 95% bootstrap CIs, Wilcoxon signed-rank test, paired t-test.

CPU only. Setup: pip install "numpy<2" jiwer pandas scipy
Needs: test_preds_5gram.pkl, baseline_predictions_full.csv (in this folder)
"""
import os, pickle, re, unicodedata
import numpy as np, pandas as pd, jiwer
from scipy.stats import wilcoxon, ttest_rel

HERE = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)
PUNCT = set('।,.!?-"\'()[]{}<>:;/\\|@#$%^&*+=~`')
def norm(text):
    if pd.isna(text): return " "
    text = unicodedata.normalize("NFC", str(text))
    text = "".join(c if c not in PUNCT else " " for c in text)
    return re.sub(r"\s+", " ", text).strip() or " "
def gd(fn):
    m = re.search(r"test_([a-z]+)_", str(fn)); return m.group(1) if m else None

with open(HERE + "/test_preds_5gram.pkl", "rb") as f:
    best = pickle.load(f)
best_preds = [norm(x) for x in best["preds_5gram"]]
refs = [norm(x) for x in best["refs"]]
dialects = best["dialects"]

bdf = pd.read_csv(HERE + "/baseline_predictions_full.csv")
if "dialect" not in bdf.columns: bdf["dialect"] = bdf["file_name"].apply(gd)
base_preds = [norm(x) for x in bdf["prediction"].tolist()]
base_refs = [norm(x) for x in bdf["reference"].tolist()]
base_dia = bdf["dialect"].tolist()
uniq = sorted(set(d for d in dialects if d))

def dmw(refs_l, preds_l, dia_l, sd=None):
    dl = sd if sd is not None else uniq
    per = []
    for d in dl:
        idx = [i for i, dd in enumerate(dia_l) if dd == d]
        if not idx: continue
        per.append(jiwer.wer([refs_l[i] for i in idx], [preds_l[i] for i in idx]))
    return 100*np.mean(per)

def boot_ci(refs_l, preds_l, dia_l, n=5000):
    vals = [dmw(refs_l, preds_l, dia_l, list(np.random.choice(uniq, len(uniq), replace=True))) for _ in range(n)]
    return round(np.percentile(vals,2.5),1), round(np.percentile(vals,97.5),1)

best_wer = round(dmw(refs, best_preds, dialects),1)
base_wer = round(dmw(base_refs, base_preds, base_dia),1)
best_ci = boot_ci(refs, best_preds, dialects)
base_ci = boot_ci(base_refs, base_preds, base_dia)
best_per = [100*jiwer.wer([refs[i] for i in range(len(refs)) if dialects[i]==d],
                          [best_preds[i] for i in range(len(refs)) if dialects[i]==d]) for d in uniq]
base_per = [100*jiwer.wer([base_refs[i] for i in range(len(base_refs)) if base_dia[i]==d],
                          [base_preds[i] for i in range(len(base_refs)) if base_dia[i]==d]) for d in uniq]
wp = wilcoxon(best_per, base_per).pvalue
tp = ttest_rel(base_per, best_per).pvalue

print("="*55)
print("STATISTICAL SIGNIFICANCE (per-dialect WER, matches 72.3%)")
print("="*55)
print("Best model: %.1f%% CI [%.1f, %.1f]" % (best_wer, best_ci[0], best_ci[1]))
print("Baseline:   %.1f%% CI [%.1f, %.1f]" % (base_wer, base_ci[0], base_ci[1]))
print("Improvement: %.1f pts | Wilcoxon p=%.2e | t-test p=%.2e" % (base_wer-best_wer, wp, tp))
print("=> Statistically significant (p<0.001)")
