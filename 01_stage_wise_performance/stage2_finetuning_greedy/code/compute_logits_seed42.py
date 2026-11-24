"""
Compute and save CTC logits of the seed-42 model for valid and test splits.
Output: valid_logits.pkl, test_logits.pkl (used by all decoding scripts).
"""

import torch, librosa, pickle, numpy as np, pandas as pd, os
from transformers import AutoProcessor, AutoModelForCTC
from tqdm import tqdm

MODEL_PATH = "/content/drive/MyDrive/indicwav2vec_regspeech12_ft/FINAL_MODEL"
DATA = "/content/drive/MyDrive/dataset"
HERE = os.path.dirname(os.path.abspath(__file__))

processor = AutoProcessor.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")
model = AutoModelForCTC.from_pretrained(MODEL_PATH, local_files_only=True).to("cuda").eval()

for split in ["test", "valid"]:
    df = pd.read_excel(f"{DATA}/{split}.xlsx")
    df["audio_path"] = df["file_name"].apply(lambda x: f"{DATA}/{split}/{x.strip()}")
    all_logits, all_refs = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
        audio, _ = librosa.load(row["audio_path"], sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits.cpu().numpy()[0].astype(np.float16)
        all_logits.append(logits)
        all_refs.append(row["transcripts"])
    with open(f"{HERE}/{split}_logits.pkl", "wb") as f:
        pickle.dump({"logits": all_logits, "refs": all_refs}, f)
    print(f"Saved {split}_logits.pkl ({len(all_logits)} samples)")
