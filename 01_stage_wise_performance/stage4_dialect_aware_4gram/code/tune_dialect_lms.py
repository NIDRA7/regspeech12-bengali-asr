"""
Tune alpha, beta and beam width separately for each dialect LM on that dialect's VALIDATION utterances.
Output: tune_dialect_results.json, tuned_dialect.json
"""

import os, pickle, json, re, unicodedata, warnings, logging, time
import numpy as np, pandas as pd, jiwer, kenlm
from pyctcdecode import build_ctcdecoder
from transformers import AutoProcessor

warnings.filterwarnings("ignore"); logging.getLogger("pyctcdecode").setLevel(logging.ERROR)
P  = "/content/drive/MyDrive/claudefinetuneandklmn"
LM = "/content/drive/MyDrive/dataset/lm_train_only"

processor = AutoProcessor.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")
vocab = sorted(processor.tokenizer.get_vocab().items(), key=lambda x: x[1])
vocab_list = [" " if t == processor.tokenizer.word_delimiter_token else t for t, _ in vocab]

vd = pickle.load(open(f"{P}/valid_logits.pkl", "rb"))
valid_df = pd.read_excel("/content/drive/MyDrive/dataset/valid.xlsx")
assert len(valid_df) == len(vd["logits"]), "valid.xlsx and logits differ in length"
valid_df["dialect"] = valid_df["file_name"].str.split("_").str[1]

def norm(t):
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub(r"[^\u0980-\u09FF\s]", " ", t).replace("।", " ")
    return re.sub(r"\s+", " ", t).strip()
refs_all = [norm(r) for r in vd["refs"]]
logits_all = [l.astype(np.float32) for l in vd["logits"]]

RES = f"{LM}/tune_dialect_results.json"
res = json.load(open(RES)) if os.path.exists(RES) else {}

def score(d, lm_path, a, b, beam, tag):
    k = f"{d}|{tag}|{a}|{b}|{beam}"
    if k not in res:
        t = time.time()
        idx = valid_df.index[valid_df["dialect"] == d].tolist()
        dec = build_ctcdecoder(labels=vocab_list, kenlm_model_path=lm_path, alpha=a, beta=b)
        preds = [norm(dec.decode(logits_all[i], beam_width=beam)) for i in idx]
        res[k] = 100 * jiwer.wer([refs_all[i] for i in idx], preds)
        json.dump(res, open(RES, "w"))
        print(f"{d:12s} {tag:7s} a={a} b={b} beam={beam}  WER={res[k]:.2f}  ({time.time()-t:.0f}s)", flush=True)
    return res[k]

best = {}
for d in sorted(valid_df["dialect"].unique()):
    lmp = f"{LM}/dialect10x_{d}.bin"
    grid = {(a, b): score(d, lmp, a, b, 50, "dialect") for a in (0.3, 0.5) for b in (0.0, 1.0, 2.0)}
    ba, bb = min(grid, key=grid.get)
    beams = {bm: score(d, lmp, ba, bb, bm, "dialect") for bm in (50, 100, 200)}
    bbm = min(beams, key=beams.get)
    g = score(d, f"{LM}/global4_clean5x.bin", 0.3, 0.0, 200, "global")
    best[d] = {"dialect": {"alpha": ba, "beta": bb, "beam": bbm, "valid_wer": beams[bbm]},
               "global":  {"alpha": 0.3, "beta": 0.0, "beam": 200, "valid_wer": g},
               "winner": "dialect" if beams[bbm] < g else "global"}
    print("==>", d, best[d], flush=True)

json.dump(best, open(f"{LM}/tuned_dialect.json", "w"), indent=2)
print("ALL DONE")
