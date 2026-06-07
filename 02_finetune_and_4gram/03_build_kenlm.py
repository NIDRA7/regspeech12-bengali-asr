"""
03_build_kenlm.py
Builds the 4-gram KenLM used for the 80.3% WER result.
Trained on RegSpeech12 train+valid transcripts, 5x upweighted.

Requires KenLM compiled at /content/kenlm_src/build/bin/
"""

import pandas as pd, re, unicodedata, os

BASE = "/content/drive/MyDrive/dataset"
OUT_DIR = "/content/drive/MyDrive/claudefinetuneandklmn/finetuneandklmn"
KENLM = "/content/kenlm_src/build/bin"

# Build KenLM if not present
if not os.path.exists(f"{KENLM}/lmplz"):
    os.system("apt-get install -y build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev")
    os.system("git clone https://github.com/kpu/kenlm.git /content/kenlm_src")
    os.system("cd /content/kenlm_src && mkdir -p build && cd build && cmake .. && make -j 4")

train_df = pd.read_excel(f"{BASE}/train.xlsx")
valid_df = pd.read_excel(f"{BASE}/valid.xlsx")
texts = list(train_df["transcripts"]) + list(valid_df["transcripts"])
texts = [str(t).strip() for t in texts if pd.notna(t)]

def clean(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\u0980-\u09FF\s।]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

corpus_path = "/content/corpus.txt"
with open(corpus_path, "w", encoding="utf-8") as f:
    for _ in range(5):  # 5x upweight
        for line in texts:
            c = clean(line)
            if len(c.split()) >= 2:
                f.write(c + "\n")

os.system(f"{KENLM}/lmplz -o 4 --discount_fallback < {corpus_path} > /content/lm_4gram.arpa")
os.system(f"{KENLM}/build_binary /content/lm_4gram.arpa {OUT_DIR}/lm_4gram.bin")
print(f"Built KenLM at {OUT_DIR}/lm_4gram.bin")
