"""
Tune alpha, beta and beam width on the VALIDATION set for the global 5-gram LM.
Output: tuned_global5.json
"""

import os, pickle, json, re, unicodedata, warnings, logging, time
import numpy as np, jiwer, kenlm
from pyctcdecode import build_ctcdecoder
from transformers import AutoProcessor

warnings.filterwarnings("ignore"); logging.getLogger("pyctcdecode").setLevel(logging.ERROR)
P  = "/content/drive/MyDrive/claudefinetuneandklmn"
LM = "/content/drive/MyDrive/dataset/lm_train_only"

processor = AutoProcessor.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")
vocab = sorted(processor.tokenizer.get_vocab().items(), key=lambda x: x[1])
vocab_list = [" " if t == processor.tokenizer.word_delimiter_token else t for t, _ in vocab]

vd = pickle.load(open(f"{P}/valid_logits.pkl", "rb"))
logits = [l.astype(np.float32) for l in vd["logits"]]

def norm(t):
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub(r"[^\u0980-\u09FF\s]", " ", t).replace("।", " ")
    return re.sub(r"\s+", " ", t).strip()
refs = [norm(r) for r in vd["refs"]]

LMPATH = f"{LM}/global5_clean5x.bin"
RES = f"{LM}/tune_global5_results.json"
res = json.load(open(RES)) if os.path.exists(RES) else {}

def get(a, b, beam):
    k = f"{a}|{b}|{beam}"
    if k not in res:
        t = time.time()
        d = build_ctcdecoder(labels=vocab_list, kenlm_model_path=LMPATH, alpha=a, beta=b)
        preds = [norm(d.decode(l, beam_width=beam)) for l in logits]
        res[k] = 100 * jiwer.wer(refs, preds)
        json.dump(res, open(RES, "w"))
        print(f"alpha={a} beta={b} beam={beam}  valid WER={res[k]:.2f}  ({time.time()-t:.0f}s)", flush=True)
    return res[k]

grid = {(a, b): get(a, b, 50) for a in (0.2, 0.3, 0.5) for b in (0.0, 1.0, 1.5, 2.0)}
ba, bb = min(grid, key=grid.get)
beams = {beam: get(ba, bb, beam) for beam in (50, 100, 200)}
bbeam = min(beams, key=beams.get)

out = {"lm": "global5_clean5x", "alpha": ba, "beta": bb, "beam": bbeam, "valid_wer": beams[bbeam]}
json.dump(out, open(f"{LM}/tuned_global5.json", "w"), indent=2)
print("BEST:", out)
