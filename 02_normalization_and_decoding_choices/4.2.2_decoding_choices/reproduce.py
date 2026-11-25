"""Reproduce Table 5: factorial ablation (LM, n-gram order, dialect-awareness, beam width) under raw / standard / BnUnicode.
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
import numpy as np
P = os.path.join(HERE, "predictions")
EXPECT = [  # (label, beam, folder, prefix, {norm: (WER, CER)})
 ("Global 4-gram", 100, os.path.join(P, "beam100"), "global4", {"raw": (78.3, 43.9), "standard": (72.3, 41.7), "bnunicode": (72.2, 41.2)}),
 ("Global 4-gram", 200, os.path.join(S1, "stage3_global_4gram", "predictions"), "global4", {"raw": (78.2, 43.1), "standard": (72.0, 40.9), "bnunicode": (71.9, 40.3)}),
 ("Dialect-aware 4-gram", 100, os.path.join(P, "beam100"), "dialect4", {"raw": (78.8, 43.8), "standard": (73.1, 41.6), "bnunicode": (72.8, 41.0)}),
 ("Dialect-aware 4-gram", 200, os.path.join(P, "dialect_beam200"), "dialect4", {"raw": (78.7, 43.1), "standard": (72.9, 40.9), "bnunicode": (72.6, 40.3)}),
 ("Global 5-gram", 100, os.path.join(P, "beam100"), "global5", {"raw": (78.4, 44.2), "standard": (72.4, 41.9), "bnunicode": (72.3, 41.4)}),
 ("Global 5-gram", 200, os.path.join(S1, "stage5_global_5gram_final", "predictions"), "global5", {"raw": (78.1, 43.3), "standard": (71.9, 41.1), "bnunicode": (71.8, 40.5)}),
]
ok = True; out = []
for label, beam, folder, prefix, exp in EXPECT:
    refs, preds, dia = load_dir(folder, prefix)
    for n in ("raw", "standard", "bnunicode"):
        r = score_by_dialect(refs, preds, dia, n)
        out.append({"lm": label, "beam": beam, "norm": n, "macro_wer": r["macro_wer"], "macro_cer": r["macro_cer"], "micro_wer": r["micro_wer"]})
        ok &= check_pair(f"{label}, beam {beam} / {n}", (r["macro_wer"], r["macro_cer"]), exp[n])
G = json.load(open(os.path.join(S1, "stage2_finetuning_greedy", "results", "stage2_finetuning_greedy_scores.json")))
for n, k in (("raw", "none"), ("standard", "standard"), ("bnunicode", "bnunicode")):
    w = np.mean(list(G[k]["wer_per_dialect"].values())); c = np.mean(list(G[k]["cer_per_dialect"].values()))
    ok &= check_pair(f"Greedy (stored scores) / {n}", (w, c), {"raw": (82.5, 42.1), "standard": (77.7, 40.2), "bnunicode": (77.4, 39.9)}[n])
json.dump(out, open(os.path.join(HERE, "results", "table5_recomputed.json"), "w"), indent=1)
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
