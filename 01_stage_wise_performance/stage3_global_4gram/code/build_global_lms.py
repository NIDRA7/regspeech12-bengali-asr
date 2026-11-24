"""
Build global 4-gram and 5-gram KenLM models from TRAIN transcripts only (cleaned text repeated 5x).
Requires compiled KenLM (lmplz, build_binary).
"""

import os, re, unicodedata, subprocess
import pandas as pd

K = "/content/kenlm/build/bin"
B = "/content/drive/MyDrive/dataset"
OUT = f"{B}/lm_train_only"
os.makedirs(OUT, exist_ok=True)
assert os.path.exists(f"{K}/lmplz"), "KenLM missing - run the install cell"

tr = pd.read_excel(f"{B}/train.xlsx")            # train only, no validation
tr["dialect"] = tr["file_name"].str.split("_").str[1]

def clean(t):
    t = unicodedata.normalize("NFC", str(t).strip())
    t = re.sub(r"[^\u0980-\u09FF\s।]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def build(texts, order, name):
    lines = [l for l in (clean(t) for t in texts if pd.notna(t)) if len(l.split()) >= 2]
    corpus = f"{OUT}/{name}.txt"
    with open(corpus, "w", encoding="utf-8") as f:
        for _ in range(5):                        # cleaned text repeated 5x
            for l in lines: f.write(l + "\n")
    subprocess.run(f"{K}/lmplz -o {order} --discount_fallback < {corpus} > {OUT}/{name}.arpa", shell=True, check=True)
    subprocess.run(f"{K}/build_binary {OUT}/{name}.arpa {OUT}/{name}.bin", shell=True, check=True)
    print(name, len(lines) * 5, "lines", flush=True)

build(tr["transcripts"], 4, "global4_clean5x")
build(tr["transcripts"], 5, "global5_clean5x")
print("ALL DONE")
