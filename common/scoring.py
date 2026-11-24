"""Shared scoring code (identical to the notebook): normalizations, per-dialect WER/CER, macro mean, population SD.
Requires: pip install jiwer numpy pandas ; pip install --no-deps bnunicodenormalizer"""
import re, unicodedata
import numpy as np, jiwer
from bnunicodenormalizer import Normalizer

PUNCT = set("।,.!?-\"'()[]{}<>:;/\\|@#$%^&*+=~`")

def std(t):
    """Standard normalization: Unicode NFC + punctuation removal + whitespace collapse."""
    if t is None or (isinstance(t, float) and np.isnan(t)): return " "
    t = unicodedata.normalize("NFC", str(t))
    t = "".join(c if c not in PUNCT else " " for c in t)
    return re.sub(r"\s+", " ", t).strip() or " "

_bn = Normalizer()
def bn(t):
    """BnUnicodeNormalizer (applied word by word) + punctuation removal."""
    if t is None or (isinstance(t, float) and np.isnan(t)) or not str(t).strip(): return " "
    t = re.sub(r"[।,.!?\-\"'()\[\]{}<>:;/\\|@#$%^&*+=~`]", " ", str(t))
    w = []
    for x in t.split():
        try:
            n = _bn(x)["normalized"]
            if n: w.append(n)
        except Exception:
            w.append(x)
    return " ".join(w).strip() or " "

NORMS = {"raw": None, "standard": std, "bnunicode": bn}

def score_by_dialect(refs, preds, dialects, norm):
    """refs, preds, dialects: equal-length lists. Returns macro WER/CER, population SD of WER, per-dialect values."""
    f = NORMS[norm]
    R = [f(r) if f else str(r) for r in refs]
    P = [f(p) if f else (p if isinstance(p, str) and p.strip() else " ") for p in preds]
    wer, cer = {}, {}
    for d in sorted(set(dialects)):
        ix = [i for i, x in enumerate(dialects) if x == d]
        wer[d] = 100 * jiwer.wer([R[i] for i in ix], [P[i] for i in ix])
        cer[d] = 100 * jiwer.cer([R[i] for i in ix], [P[i] for i in ix])
    w = np.array(list(wer.values()))
    return {"macro_wer": float(w.mean()), "macro_cer": float(np.mean(list(cer.values()))),
            "sd_wer": float(w.std(ddof=0)),   # population SD, as in Tables 3 and 6
            "micro_wer": 100 * jiwer.wer(R, P), "wer_per_dialect": wer, "cer_per_dialect": cer}

def check(name, got, expected, tol=0.06):
    """Compare a (WER, CER, SD) triple with the paper values; print PASS/FAIL."""
    ok = all(abs(g - e) <= tol + 0.05 for g, e in zip(got, expected))
    print(f"{name:<22} got WER/CER/SD = {got[0]:.1f}/{got[1]:.1f}/{got[2]:.1f}   paper = {expected[0]}/{expected[1]}/{expected[2]}   {'PASS' if ok else 'FAIL'}")
    return ok
