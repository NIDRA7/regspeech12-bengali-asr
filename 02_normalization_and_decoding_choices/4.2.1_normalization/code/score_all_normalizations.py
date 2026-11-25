"""
Score all stages under raw, standard and BnUnicode normalization (WER and CER, macro per dialect).
Output: final_scores_all_norms.json
"""

import subprocess, sys, re, pickle, json, unicodedata
def pipi(a): subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + a)
try: import jiwer
except ImportError: pipi(["jiwer"]); import jiwer
try: from bnunicodenormalizer import Normalizer
except ImportError: pipi(["--no-deps", "bnunicodenormalizer"]); from bnunicodenormalizer import Normalizer
import numpy as np, pandas as pd
from transformers import AutoProcessor

P  = "/content/drive/MyDrive/claudefinetuneandklmn"
LM = "/content/drive/MyDrive/dataset/lm_train_only"
PR = f"{LM}/test_preds"

td = pickle.load(open(f"{P}/test_logits.pkl", "rb"))
refs_all = [str(r) for r in td["refs"]]
test_df = pd.read_excel("/content/drive/MyDrive/dataset/test.xlsx")
test_df["dialect"] = test_df["file_name"].str.split("_").str[1]
dialects = sorted(test_df["dialect"].unique())
didx = {d: test_df.index[test_df["dialect"] == d].tolist() for d in dialects}

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

processor = AutoProcessor.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")
greedy = [processor.batch_decode(np.argmax(l.astype(np.float32), axis=-1)[None], skip_special_tokens=True)[0]
          for l in td["logits"]]
greedy = [p if p.strip() else " " for p in greedy]

def load_stage(stage):
    preds = [None] * len(refs_all)
    for d in dialects:
        o = pickle.load(open(f"{PR}/{stage}_{d}.pkl", "rb"))
        for i, p in zip(o["idx"], o["preds"]): preds[i] = p
    return [p if p.strip() else " " for p in preds]

stages = {"Greedy": greedy, "Global 4-gram": load_stage("global4"),
          "Dialect-aware 4-gram": load_stage("dialect4"), "Global 5-gram": load_stage("global5")}
norms = {"none": None, "standard": std, "bnunicode": bn}
R = {n: ([f(r) for r in refs_all] if f else refs_all) for n, f in norms.items()}

def score(preds, n):
    f = norms[n]; Rn = R[n]; Pp = [f(p) for p in preds] if f else preds
    wer = [100 * jiwer.wer([Rn[i] for i in ix], [Pp[i] for i in ix]) for ix in didx.values()]
    cer = [100 * jiwer.cer([Rn[i] for i in ix], [Pp[i] for i in ix]) for ix in didx.values()]
    return {"macro_wer": float(np.mean(wer)), "macro_cer": float(np.mean(cer)),
            "micro_wer": 100 * jiwer.wer(Rn, Pp), "wer_per_dialect": dict(zip(didx, wer)),
            "cer_per_dialect": dict(zip(didx, cer))}

res = {}
for s, p in stages.items():
    res[s] = {n: score(p, n) for n in norms}
    print("scored", s, flush=True)

print("\nTABLE 1: all stages, BnUnicodeNormalizer")
print(f"{'System':<24}{'macro WER':>10}{'micro WER':>10}{'macro CER':>10}")
for s in stages:
    r = res[s]["bnunicode"]
    print(f"{s:<24}{r['macro_wer']:>10.1f}{r['micro_wer']:>10.1f}{r['macro_cer']:>10.1f}")

print("\nTABLE 2: macro WER (CER) by normalization")
print(f"{'System':<24}{'none':>16}{'standard':>16}{'bnunicode':>16}")
for s in stages:
    cells = [f"{res[s][n]['macro_wer']:.1f} ({res[s][n]['macro_cer']:.1f})" for n in norms]
    print(f"{s:<24}{cells[0]:>16}{cells[1]:>16}{cells[2]:>16}")

json.dump(res, open(f"{LM}/final_scores_all_norms.json", "w"), indent=2)
print("\nsaved final_scores_all_norms.json")
