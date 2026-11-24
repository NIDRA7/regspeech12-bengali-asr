"""
Decode the TEST set once with the tuned settings (global 4-gram, dialect-aware 4-gram, global 5-gram).
Output: test_preds/{stage}_{dialect}.pkl
"""

import os, pickle, json, warnings, logging, time
import numpy as np, pandas as pd, kenlm
from pyctcdecode import build_ctcdecoder
from transformers import AutoProcessor

warnings.filterwarnings("ignore"); logging.getLogger("pyctcdecode").setLevel(logging.ERROR)
P  = "/content/drive/MyDrive/claudefinetuneandklmn"
LM = "/content/drive/MyDrive/dataset/lm_train_only"
OUT = f"{LM}/test_preds"; os.makedirs(OUT, exist_ok=True)

processor = AutoProcessor.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")
vocab = sorted(processor.tokenizer.get_vocab().items(), key=lambda x: x[1])
vocab_list = [" " if t == processor.tokenizer.word_delimiter_token else t for t, _ in vocab]

td = pickle.load(open(f"{P}/test_logits.pkl", "rb"))
logits = [l.astype(np.float32) for l in td["logits"]]
refs_raw = list(td["refs"])
test_df = pd.read_excel("/content/drive/MyDrive/dataset/test.xlsx")
assert len(test_df) == len(logits), "test.xlsx and logits differ in length"
test_df["dialect"] = test_df["file_name"].str.split("_").str[1]
dialects = sorted(test_df["dialect"].unique())
print(len(logits), "test utterances;", len(dialects), "dialects", flush=True)

def decode_stage(stage, d, dec, beam):
    f = f"{OUT}/{stage}_{d}.pkl"
    if os.path.exists(f):
        return
    t = time.time()
    idx = test_df.index[test_df["dialect"] == d].tolist()
    preds = [dec.decode(logits[i], beam_width=beam) for i in idx]
    pickle.dump({"idx": idx, "preds": preds, "refs": [refs_raw[i] for i in idx]}, open(f, "wb"))
    print(f"{stage:9s} {d:12s} n={len(idx):4d}  ({time.time()-t:.0f}s)", flush=True)

# Stage 3 and Stage 6: global LMs, settings tuned on validation
for stage, lmfile in (("global4", "global4_clean5x.bin"), ("global5", "global5_clean5x.bin")):
    dec = build_ctcdecoder(labels=vocab_list, kenlm_model_path=f"{LM}/{lmfile}", alpha=0.3, beta=0.0)
    for d in dialects:
        decode_stage(stage, d, dec, 200)

# Stage 4: dialect-aware LMs, per-dialect settings tuned on validation
tuned = json.load(open(f"{LM}/tuned_dialect.json"))
for d in dialects:
    s = tuned[d]["dialect"]
    dec = build_ctcdecoder(labels=vocab_list, kenlm_model_path=f"{LM}/dialect10x_{d}.bin",
                           alpha=s["alpha"], beta=s["beta"])
    decode_stage("dialect4", d, dec, s["beam"])

print("ALL DONE")
