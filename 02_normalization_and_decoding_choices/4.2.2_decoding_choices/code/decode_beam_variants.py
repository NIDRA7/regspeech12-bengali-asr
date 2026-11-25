"""
Decode the test set at beam 100 (all LMs) and the dialect-aware LMs at beam 200, for the factorial ablation.
"""

import os, pickle, json, warnings, logging, time
import numpy as np, pandas as pd, kenlm
from pyctcdecode import build_ctcdecoder
from transformers import AutoProcessor

warnings.filterwarnings("ignore"); logging.getLogger("pyctcdecode").setLevel(logging.ERROR)
P  = "/content/drive/MyDrive/claudefinetuneandklmn"
LM = "/content/drive/MyDrive/dataset/lm_train_only"
O100  = f"{LM}/test_preds_beam100";          os.makedirs(O100, exist_ok=True)
O200D = f"{LM}/test_preds_dialect_beam200";  os.makedirs(O200D, exist_ok=True)

processor = AutoProcessor.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")
vocab = sorted(processor.tokenizer.get_vocab().items(), key=lambda x: x[1])
vocab_list = [" " if t == processor.tokenizer.word_delimiter_token else t for t, _ in vocab]

td = pickle.load(open(f"{P}/test_logits.pkl", "rb"))
logits = [l.astype(np.float32) for l in td["logits"]]
refs_raw = list(td["refs"])
test_df = pd.read_excel("/content/drive/MyDrive/dataset/test.xlsx")
test_df["dialect"] = test_df["file_name"].str.split("_").str[1]
dialects = sorted(test_df["dialect"].unique())
tuned = json.load(open(f"{LM}/tuned_dialect.json"))

def run(stage, d, dec, beam, outdir):
    f = f"{outdir}/{stage}_{d}.pkl"
    if os.path.exists(f): return
    t = time.time()
    idx = test_df.index[test_df["dialect"] == d].tolist()
    preds = [dec.decode(logits[i], beam_width=beam) for i in idx]
    pickle.dump({"idx": idx, "preds": preds, "refs": [refs_raw[i] for i in idx]}, open(f, "wb"))
    print(f"{stage:9s} {d:12s} beam={beam:3d}  ({time.time()-t:.0f}s)", flush=True)

# global 4-gram and 5-gram at beam 100 (beam 200 already done)
for stage, lmfile in (("global4", "global4_clean5x.bin"), ("global5", "global5_clean5x.bin")):
    dec = build_ctcdecoder(labels=vocab_list, kenlm_model_path=f"{LM}/{lmfile}", alpha=0.3, beta=0.0)
    for d in dialects:
        run(stage, d, dec, 100, O100)

# dialect-aware: each dialect's own tuned alpha/beta, fixed beam 100 and 200
for d in dialects:
    s = tuned[d]["dialect"]
    dec = build_ctcdecoder(labels=vocab_list, kenlm_model_path=f"{LM}/dialect10x_{d}.bin",
                           alpha=s["alpha"], beta=s["beta"])
    run("dialect4", d, dec, 100, O100)
    run("dialect4", d, dec, 200, O200D)

print("ALL DONE")
