"""
02_decode_5gram_and_normalize.py
Decode saved test logits with 5-gram KenLM (beam 200), normalize, compute macro Mean WER.
Produces 5-gram LM + normalization = 72.3% Mean WER.
CPU only, ~15 min.

Setup:
    pip install "numpy<2" jiwer pyctcdecode pandas openpyxl transformers
    pip install https://github.com/kpu/kenlm/archive/master.zip

Needs in this folder: test_logits.pkl, lm_5gram.bin ; and dataset test.xlsx for dialect labels.
"""
import os, pickle, re, unicodedata
import numpy as np, pandas as pd, jiwer
from pyctcdecode import build_ctcdecoder
from transformers import AutoProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/content/drive/MyDrive/dataset"

processor = AutoProcessor.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")
vd = processor.tokenizer.get_vocab()
vocab_list = [t if t != processor.tokenizer.word_delimiter_token else " "
              for t, _ in sorted(vd.items(), key=lambda x: x[1])]

PUNCT = '।,.!?-"\'()[]{}<>:;/\\|@#$%^&*+=~`'
PUNCT_SET = set(PUNCT)
def normalize(text):
    if pd.isna(text):
        return " "
    text = unicodedata.normalize("NFC", str(text))
    text = "".join(c if c not in PUNCT_SET else " " for c in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else " "

def get_dialect(fn):
    m = re.search(r"test_([a-z]+)_", str(fn))
    return m.group(1) if m else None

with open(HERE + "/test_logits.pkl", "rb") as f:
    td = pickle.load(f)
test_df = pd.read_excel(DATA + "/test.xlsx")
dialects = [get_dialect(fn) for fn in test_df["file_name"].tolist()]

decoder = build_ctcdecoder(vocab_list, kenlm_model_path=HERE + "/lm_5gram.bin", alpha=0.5, beta=1.5)
preds = [decoder.decode(lg.astype(np.float32), beam_width=200) for lg in td["logits"]]

per = {}
for d in sorted(set(x for x in dialects if x)):
    idx = [i for i, dd in enumerate(dialects) if dd == d]
    r = [normalize(td["refs"][i]) for i in idx]
    p = [normalize(preds[i]) for i in idx]
    per[d] = round(100 * jiwer.wer(r, p), 1)
mean = round(sum(per.values()) / len(per), 1)
for d in sorted(per, key=lambda x: per[x]):
    print(d, per[d])
print("MEAN WER:", mean)
