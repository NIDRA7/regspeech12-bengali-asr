"""
Factorial ablation: LM, n-gram order, dialect-awareness, beam width, under three scorings.
Output: factorial_ablation.json
"""

import subprocess, sys, os, re, pickle, json, unicodedata
def pipi(a): subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + a)
try: import jiwer
except ImportError: pipi(["jiwer"]); import jiwer
try: from bnunicodenormalizer import Normalizer
except ImportError: pipi(["--no-deps", "bnunicodenormalizer"]); from bnunicodenormalizer import Normalizer
import numpy as np, pandas as pd
from transformers import AutoProcessor

P  = "/content/drive/MyDrive/claudefinetuneandklmn"
LM = "/content/drive/MyDrive/dataset/lm_train_only"
DIRS = {("global4", 200): f"{LM}/test_preds", ("global5", 200): f"{LM}/test_preds",
        ("global4", 100): f"{LM}/test_preds_beam100", ("global5", 100): f"{LM}/test_preds_beam100",
        ("dialect4", 100): f"{LM}/test_preds_beam100", ("dialect4", 200): f"{LM}/test_preds_dialect_beam200"}

td = pickle.load(open(f"{P}/test_logits.pkl", "rb"))
refs_all = [str(r) for r in td["refs"]]
test_df = pd.read_excel("/content/drive/MyDrive/dataset/test.xlsx")
test_df["dialect"] = test_df["file_name"].str.split("_").str[1]
dialects = sorted(test_df["dialect"].unique())
didx = {d: test_df.index[test_df["dialect"] == d].tolist() for d in dialects}

missing = [f"{DIRS[k]}/{k[0]}_{d}.pkl" for k in DIRS for d in dialects
           if not os.path.exists(f"{DIRS[k]}/{k[0]}_{d}.pkl")]
assert not missing, f"missing {len(missing)} files, e.g. {missing[:3]}"

PUNCT = set("।,.!?-\"'()[]{}<>:;/\\|@#$%^&*+=~`")
def std(t):
    if pd.isna(t): return " "
    t = unicodedata.normalize("NFC", str(t))
    t = "".join(c if c not in PUNCT else " " for c in t)
    return re.sub(r"\s+", " ", t).strip() or " "
bnorm = Normalizer()
def bn(t):
    if pd.isna(t) or not str(t).strip(): return " "
    t = re.sub(r"[।,.!?\-\"'()\[\]{}<>:;/\\|@#$%^&*+=~`]", " ", str(t))
    w = []
    for x in t.split():
        try:
            n = bnorm(x)["normalized"]
            if n: w.append(n)
        except Exception:
            w.append(x)
    return " ".join(w).strip() or " "
norms = {"none": None, "standard": std, "bnunicode": bn}
R = {n: ([f(r) for r in refs_all] if f else refs_all) for n, f in norms.items()}

processor = AutoProcessor.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")
greedy = [processor.batch_decode(np.argmax(l.astype(np.float32), axis=-1)[None], skip_special_tokens=True)[0]
          for l in td["logits"]]
greedy = [p if p.strip() else " " for p in greedy]

def load(stage, beam):
    preds = [None] * len(refs_all)
    for d in dialects:
        o = pickle.load(open(f"{DIRS[(stage, beam)]}/{stage}_{d}.pkl", "rb"))
        for i, p in zip(o["idx"], o["preds"]): preds[i] = p
    return [p if p.strip() else " " for p in preds]

def score(preds, n):
    f = norms[n]; Rn = R[n]; Pp = [f(p) for p in preds] if f else preds
    wer = [100 * jiwer.wer([Rn[i] for i in ix], [Pp[i] for i in ix]) for ix in didx.values()]
    cer = [100 * jiwer.cer([Rn[i] for i in ix], [Pp[i] for i in ix]) for ix in didx.values()]
    return float(np.mean(wer)), 100 * jiwer.wer(Rn, Pp), float(np.mean(cer)), dict(zip(didx, wer)), dict(zip(didx, cer))

rows = [("Fine-tuned, greedy (no LM)", "-", n, greedy) for n in ("none", "standard", "bnunicode")]
for label, stage in (("Global 4-gram", "global4"), ("Dialect-aware 4-gram", "dialect4"), ("Global 5-gram", "global5")):
    p100, p200 = load(stage, 100), load(stage, 200)
    rows += [(label, b, n, p) for b, p in ((100, p100), (200, p200))
             for n in ("none", "standard", "bnunicode")]

out = []
print("| LM | Beam | Normalization | Macro WER | Micro WER | Macro CER |")
print("|---|---|---|---|---|---|")
for label, beam, n, preds in rows:
    mw, uw, mc, wd, cd = score(preds, n)
    print(f"| {label} | {beam} | {n} | {mw:.1f} | {uw:.1f} | {mc:.1f} |", flush=True)
    out.append({"lm": label, "beam": beam, "norm": n, "macro_wer": mw, "micro_wer": uw,
                "macro_cer": mc, "wer_per_dialect": wd, "cer_per_dialect": cd})

json.dump(out, open(f"{LM}/factorial_ablation.json", "w"), indent=2)
print("\nsaved factorial_ablation.json")
