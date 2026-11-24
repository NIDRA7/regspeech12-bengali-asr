"""Reproduce the Stage 3: Fine-tuning + global 4-gram LM (tested) row of Table 3 (Raw and BnUnicode: WER, CER, SD).
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
for name, norm, exp in (("raw", "raw", (78.2, 43.1, 12.1)), ("bnunicode", "bnunicode", (71.9, 40.3, 14.2))):
    r = score_by_dialect(refs, preds, dia, norm)
    ok &= check("Stage 3 " + name, (r["macro_wer"], r["macro_cer"], r["sd_wer"]), exp)
sys.exit(0 if ok else 1)
