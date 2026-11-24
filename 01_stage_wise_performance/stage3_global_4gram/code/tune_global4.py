"""
Tune alpha, beta (at beam 50) and then beam width on the VALIDATION set for the global 4-gram LM.
Output: tuned_global4.json
"""

import pickle, json, re, unicodedata, warnings, logging, time
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
print(len(logits), "validation utterances", flush=True)

def norm(t):
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub(r"[^\u0980-\u09FF\s]", " ", t).replace("।", " ")
    return re.sub(r"\s+", " ", t).strip()
refs = [norm(r) for r in vd["refs"]]

def run(lm_path, alpha, beta, beam):
    d = build_ctcdecoder(labels=vocab_list, kenlm_model_path=lm_path, alpha=alpha, beta=beta)
    preds = [norm(d.decode(l, beam_width=beam)) for l in logits]
    return 100 * jiwer.wer(refs, preds)

LMPATH = f"{LM}/global4_clean5x.bin"

res = {}
for a in (0.3, 0.5, 0.8):
    for b in (0.0, 1.0, 1.5, 2.0):
        t = time.time(); res[(a, b)] = run(LMPATH, a, b, 50)
        print(f"alpha={a} beta={b} beam=50  valid WER={res[(a,b)]:.2f}  ({time.time()-t:.0f}s)", flush=True)
ba, bb = min(res, key=res.get)

beams = {}
for beam in (50, 100, 200):
    beams[beam] = run(LMPATH, ba, bb, beam)
    print(f"best alpha={ba} beta={bb} beam={beam}  valid WER={beams[beam]:.2f}", flush=True)
bbeam = min(beams, key=beams.get)

out = {"lm": "global4_clean5x", "alpha": ba, "beta": bb, "beam": bbeam, "valid_wer": beams[bbeam]}
json.dump(out, open(f"{LM}/tuned_global4.json", "w"), indent=2)
print("BEST:", out)
