"""Reproduce Table 4: effect of text normalization (raw / standard / BnUnicode) on WER and CER at every stage.
Usage: python reproduce.py   (needs folder 01_stage_wise_performance in the repo)"""
import sys, os, glob, json, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "common"))
from scoring import score_by_dialect
S1 = os.path.join(REPO, "01_stage_wise_performance")

def load_dir(folder, prefix):
    refs, preds, dia = [], [], []
    for f in sorted(glob.glob(os.path.join(folder, prefix + "_*.pkl"))):
        o = pickle.load(open(f, "rb"))
        dname = os.path.basename(f).replace(".pkl", "").split("_", 1)[1]
        refs += [str(r) for r in o["refs"]]; preds += o["preds"]; dia += [dname] * len(o["preds"])
    return refs, preds, dia

def check_pair(name, got, exp, tol=0.1):
    ok = abs(got[0] - exp[0]) <= tol and abs(got[1] - exp[1]) <= tol
    print(f"{name:<44} WER/CER = {got[0]:.1f}/{got[1]:.1f}   paper = {exp[0]}/{exp[1]}   {'PASS' if ok else 'FAIL'}")
    return ok
import numpy as np, pandas as pd
EXPECT = {  # system: {norm: (WER, CER)}
 "Zero-shot IndicWav2Vec": {"raw": (89.0, 60.1), "standard": (86.8, 58.4), "bnunicode": (86.6, 58.7)},
 "Fine-tuned, greedy": {"raw": (82.5, 42.1), "standard": (77.7, 40.2), "bnunicode": (77.4, 39.9)},
 "+ global 4-gram LM": {"raw": (78.2, 43.1), "standard": (72.0, 40.9), "bnunicode": (71.9, 40.3)},
 "+ dialect-aware 4-gram LMs": {"raw": (78.8, 43.4), "standard": (73.0, 41.3), "bnunicode": (72.7, 40.6)},
 "+ global 5-gram LM": {"raw": (78.1, 43.3), "standard": (71.9, 41.1), "bnunicode": (71.8, 40.5)},
}
ok = True
# zero-shot: from saved predictions CSV
df = pd.read_csv(os.path.join(S1, "stage1_zero_shot", "predictions", "zero_shot_predictions.csv"), keep_default_na=False)
for n in ("raw", "standard", "bnunicode"):
    r = score_by_dialect(df.reference.tolist(), df.prediction.tolist(), df.dialect.tolist(), n)
    ok &= check_pair("Zero-shot / " + n, (r["macro_wer"], r["macro_cer"]), EXPECT["Zero-shot IndicWav2Vec"][n])
# greedy: stored scores (greedy predictions need the logits file)
G = json.load(open(os.path.join(HERE, "results", "final_scores_all_norms.json")))["Greedy"]
for n, k in (("raw", "none"), ("standard", "standard"), ("bnunicode", "bnunicode")):
    ok &= check_pair("Greedy / " + n + " (stored scores)", (np.mean(list(G[k]["wer_per_dialect"].values())), np.mean(list(G[k]["cer_per_dialect"].values()))), EXPECT["Fine-tuned, greedy"][n])
# LM stages: from saved predictions of folder 01
for label, folder, prefix in (("+ global 4-gram LM", "stage3_global_4gram", "global4"),
                              ("+ dialect-aware 4-gram LMs", "stage4_dialect_aware_4gram", "dialect4"),
                              ("+ global 5-gram LM", "stage5_global_5gram_final", "global5")):
    refs, preds, dia = load_dir(os.path.join(S1, folder, "predictions"), prefix)
    for n in ("raw", "standard", "bnunicode"):
        r = score_by_dialect(refs, preds, dia, n)
        ok &= check_pair(label + " / " + n, (r["macro_wer"], r["macro_cer"]), EXPECT[label][n])
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
