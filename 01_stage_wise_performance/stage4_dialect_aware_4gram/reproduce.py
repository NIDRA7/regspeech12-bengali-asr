"""Reproduce the Stage 4: Fine-tuning + dialect-aware 4-gram LMs (tested) row of Table 3 (Raw and BnUnicode: WER, CER, SD).
Usage: python reproduce.py"""
import sys, os, glob, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from scoring import score_by_dialect, check
HERE = os.path.dirname(os.path.abspath(__file__))
refs, preds, dia = [], [], []
for f in sorted(glob.glob(os.path.join(HERE, "predictions", "*.pkl"))):
    o = pickle.load(open(f, "rb"))
    dname = os.path.basename(f).replace(".pkl", "").split("_", 1)[1]
    refs += [str(r) for r in o["refs"]]; preds += o["preds"]; dia += [dname] * len(o["preds"])
ok = True
for name, norm, exp in (("raw", "raw", (78.8, 43.4, 10.8)), ("bnunicode", "bnunicode", (72.7, 40.6, 12.6))):
    r = score_by_dialect(refs, preds, dia, norm)
    ok &= check("Stage 4 " + name, (r["macro_wer"], r["macro_cer"], r["sd_wer"]), exp)
sys.exit(0 if ok else 1)
