"""
Rebuild the Comilla and Noakhali dialect LMs as 4-grams (10x repeat) and clear their old tuning entries.
"""

import os, re, unicodedata, subprocess, json
import pandas as pd

K = "/content/kenlm_src/build/bin"
if not os.path.exists(f"{K}/lmplz"):
    print("compiling KenLM tools (about 5 min)...", flush=True)
    subprocess.run("apt-get install -y -qq cmake libboost-program-options-dev libboost-system-dev libboost-thread-dev libboost-test-dev libbz2-dev liblzma-dev zlib1g-dev && cd /content/kenlm_src && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4", shell=True, check=True)

B  = "/content/drive/MyDrive/dataset"
LM = f"{B}/lm_train_only"
tr = pd.read_excel(f"{B}/train.xlsx").dropna(subset=["transcripts"])
tr["dialect"] = tr["file_name"].str.split("_").str[1]

def clean(t):
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub(r"[^\u0980-\u09FF\s।]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

for d in ("comilla", "noakhali"):
    corpus = f"{LM}/dialect10x_{d}.txt"
    with open(corpus, "w", encoding="utf-8") as f:
        for _ in range(10):
            for line in tr[tr["dialect"] == d]["transcripts"]:
                c = clean(line)
                if len(c.split()) >= 2: f.write(c + "\n")
    subprocess.run(f"{K}/lmplz -o 4 --discount_fallback < {corpus} > {LM}/dialect10x_{d}.arpa", shell=True, check=True)
    subprocess.run(f"{K}/build_binary {LM}/dialect10x_{d}.arpa {LM}/dialect10x_{d}.bin", shell=True, check=True)
    print(d, "rebuilt as 4-gram", flush=True)

R = f"{LM}/tune_dialect_results.json"
if os.path.exists(R):
    res = json.load(open(R))
    res = {k: v for k, v in res.items() if not k.startswith(("comilla|", "noakhali|"))}
    json.dump(res, open(R, "w"))
    print("cleared old tuning for comilla and noakhali")
print("DONE")
