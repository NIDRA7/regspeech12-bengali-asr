"""
Build one 4-gram KenLM per dialect from that dialect's TRAIN transcripts only (cleaned text repeated 10x).
Note: this script builds a 3-gram for dialects with <5000 lines; the Comilla and Noakhali LMs
were then rebuilt as 4-grams by rebuild_small_dialect_lms.py so that all dialect LMs are 4-gram.
"""

import os, re, unicodedata, subprocess
import pandas as pd

K = "/content/kenlm/build/bin"
B = "/content/drive/MyDrive/dataset"
OUT = f"{B}/lm_train_only"
assert os.path.exists(f"{K}/lmplz"), "KenLM missing - run the install cell"

tr = pd.read_excel(f"{B}/train.xlsx").dropna(subset=["transcripts"])
tr["dialect"] = tr["file_name"].str.split("_").str[1]

def clean(t):
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub(r"[^\u0980-\u09FF\s।]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

for d, g in tr.groupby("dialect"):
    corpus = f"{OUT}/dialect10x_{d}.txt"
    with open(corpus, "w", encoding="utf-8") as f:
        for _ in range(10):
            for line in g["transcripts"]:
                c = clean(line)
                if len(c.split()) >= 2:
                    f.write(c + "\n")
    n = sum(1 for _ in open(corpus, encoding="utf-8"))
    order = 3 if n < 5000 else 4
    subprocess.run(f"{K}/lmplz -o {order} --discount_fallback < {corpus} > {OUT}/dialect10x_{d}.arpa", shell=True, check=True)
    subprocess.run(f"{K}/build_binary {OUT}/dialect10x_{d}.arpa {OUT}/dialect10x_{d}.bin", shell=True, check=True)
    print(d, n, "lines,", order, "-gram", flush=True)
print("ALL DONE")
