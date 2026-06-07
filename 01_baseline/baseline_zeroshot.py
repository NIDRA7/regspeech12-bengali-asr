"""
baseline_zeroshot.py
Zero-shot IndicWav2Vec baseline on RegSpeech12 test set.
Produces Mean WER: 89.0% across 12 dialects.

Requirements:
    pip install transformers librosa jiwer pandas tqdm openpyxl
    
Setup:
    - Dataset at /content/drive/MyDrive/dataset/ with test.xlsx + audio in test/
    - GPU recommended (~20 min on T4)

Usage:
    python baseline_zeroshot.py
"""

import torch, librosa, jiwer, numpy as np, pandas as pd, re, json, os
from transformers import AutoProcessor, AutoModelForCTC
from tqdm import tqdm

MODEL_NAME = "ai4bharat/indicwav2vec_v1_bengali"
DATA_PATH = "/content/drive/MyDrive/dataset"
OUT_DIR = "/content/drive/MyDrive/claudefinetuneandklmn/baselineperformance"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading IndicWav2Vec (zero-shot)...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForCTC.from_pretrained(MODEL_NAME).to("cuda").eval()

test_df = pd.read_excel(f"{DATA_PATH}/test.xlsx")
test_df["audio_path"] = test_df["file_name"].apply(lambda x: f"{DATA_PATH}/test/{x.strip()}")
test_df["dialect"] = test_df["file_name"].apply(
    lambda fn: (re.search(r"test_([a-z]+)_", str(fn)).group(1)
                if re.search(r"test_([a-z]+)_", str(fn)) else None)
)

print(f"Running zero-shot on {len(test_df)} test samples...")
preds, refs, dialects, files = [], [], [], []
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    try:
        audio, _ = librosa.load(row["audio_path"], sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    except Exception:
        pred = ""
    preds.append(pred)
    refs.append(str(row["transcripts"]))
    dialects.append(row["dialect"])
    files.append(row["file_name"])

# Save predictions
pd.DataFrame({
    "file_name": files, "dialect": dialects,
    "reference": refs, "prediction": preds,
}).to_csv(f"{OUT_DIR}/baseline_predictions_full.csv", index=False)

# Per-dialect WER (pooled jiwer)
per_dialect = {}
for d in sorted(set(dialects)):
    if d is None:
        continue
    sub_refs  = [r for r, dia in zip(refs, dialects)  if dia == d]
    sub_preds = [p if p.strip() else " " for p, dia in zip(preds, dialects) if dia == d]
    per_dialect[d] = {"n": len(sub_refs), "wer": round(100 * jiwer.wer(sub_refs, sub_preds), 1)}

mean_wer = float(np.mean([v["wer"] for v in per_dialect.values()]))

print("=" * 50)
print("BASELINE — Zero-shot IndicWav2Vec on RegSpeech12 Test")
print("=" * 50)
for d, v in sorted(per_dialect.items(), key=lambda x: x[1]["wer"]):
    print(f"  {d:<14} {v[\'n\']:>5} {v[\'wer\']:>6.1f}%")
print("-" * 30)
print(f"  Mean WER: {mean_wer:.1f}%")

with open(f"{OUT_DIR}/baseline_results.json", "w") as f:
    json.dump({
        "model": MODEL_NAME,
        "test_set_samples": len(refs),
        "mean_wer": round(mean_wer, 1),
        "per_dialect": per_dialect,
    }, f, indent=2)
print(f"\nSaved to {OUT_DIR}/")
