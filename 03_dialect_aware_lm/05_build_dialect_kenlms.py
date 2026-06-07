"""
05_build_dialect_kenlms.py
Builds a KenLM per dialect (4-gram, or 3-gram for low-data dialects).
Each dialect's transcripts are upweighted 10x in its own LM.

Output: dialect_lms/lm_<dialect>.bin for all 12 dialects.

Run: ~10-15 min total.
Requires KenLM compiled at /content/kenlm_src/build/bin/
"""
import pandas as pd, re, unicodedata, os

BASE = "/content/drive/MyDrive/dataset"
HERE = os.path.dirname(os.path.abspath(__file__))
LM_DIR = f"{HERE}/dialect_lms"
KENLM = "/content/kenlm_src/build/bin"
os.makedirs(LM_DIR, exist_ok=True)

if not os.path.exists(f"{KENLM}/lmplz"):
    os.system("apt-get install -y build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev")
    os.system("git clone https://github.com/kpu/kenlm.git /content/kenlm_src")
    os.system("cd /content/kenlm_src && mkdir -p build && cd build && cmake .. && make -j 4")

train_df = pd.read_excel(f"{BASE}/train.xlsx")
valid_df = pd.read_excel(f"{BASE}/valid.xlsx")
all_df = pd.concat([train_df, valid_df], ignore_index=True)

def get_dialect(fn):
    m = re.search(r"(?:train|valid)_([a-z]+)_", str(fn))
    return m.group(1) if m else None

all_df["dialect"] = all_df["file_name"].apply(get_dialect)
all_df = all_df.dropna(subset=["dialect", "transcripts"])

def clean(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"[^\u0980-\u09FF\s।]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

for dialect in sorted(all_df["dialect"].unique()):
    texts = all_df[all_df["dialect"] == dialect]["transcripts"].tolist()
    corpus_path = f"/content/corpus_{dialect}.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for _ in range(10):
            for line in texts:
                c = clean(line)
                if len(c.split()) >= 2:
                    f.write(c + "\n")
    lines = sum(1 for _ in open(corpus_path, encoding="utf-8"))
    order = 3 if lines < 5000 else 4
    arpa = f"/content/lm_{dialect}.arpa"
    bin_path = f"{LM_DIR}/lm_{dialect}.bin"
    os.system(f"{KENLM}/lmplz -o {order} --discount_fallback < {corpus_path} > {arpa} 2>/dev/null")
    os.system(f"{KENLM}/build_binary {arpa} {bin_path} 2>/dev/null")
    print(f"{dialect}: {lines} lines, {order}-gram -> {bin_path}")
print("Done.")
