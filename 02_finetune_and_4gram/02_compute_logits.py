"""
02_compute_logits.py
Runs fine-tuned model on test (and valid) set, saves logits to disk.
Lets you re-decode with different LMs/alphas without re-running the model.

Run on T4 GPU (~20-30 min):
    python 02_compute_logits.py
"""

import torch, librosa, pickle, numpy as np, pandas as pd, os
from transformers import AutoProcessor, AutoModelForCTC
from tqdm import tqdm

MODEL_PATH = "/content/drive/MyDrive/indicwav2vec_regspeech12_ft/FINAL_MODEL"
DATA_PATH = "/content/drive/MyDrive/dataset"
OUT_DIR = "/content/drive/MyDrive/claudefinetuneandklmn/finetuneandklmn"
os.makedirs(OUT_DIR, exist_ok=True)

processor = AutoProcessor.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")
model = AutoModelForCTC.from_pretrained(MODEL_PATH, local_files_only=True).to("cuda").eval()

for split in ["test", "valid"]:
    df = pd.read_excel(f"{DATA_PATH}/{split}.xlsx")
    df["audio_path"] = df["file_name"].apply(lambda x: f"{DATA_PATH}/{split}/{x.strip()}")
    
    all_logits, all_refs = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
        audio, _ = librosa.load(row["audio_path"], sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits.cpu().numpy()[0].astype(np.float16)
        all_logits.append(logits)
        all_refs.append(row["transcripts"])
    
    with open(f"{OUT_DIR}/{split}_logits.pkl", "wb") as f:
        pickle.dump({"logits": all_logits, "refs": all_refs}, f)
    print(f"Saved {split}_logits.pkl ({len(all_logits)} samples)")
