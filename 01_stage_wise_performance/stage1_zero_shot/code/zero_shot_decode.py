"""
Zero-shot IndicWav2Vec baseline (no fine-tuning), greedy CTC decoding on the test set.
Saves baseline_predictions_full.csv.
"""

import torch, librosa, jiwer, numpy as np, pandas as pd, re, os
from transformers import AutoModelForCTC
from tqdm import tqdm

# Load BASE model (zero-shot — no fine-tuning)
print("Loading base IndicWav2Vec...")
base_model = AutoModelForCTC.from_pretrained("ai4bharat/indicwav2vec_v1_bengali").to("cuda").eval()

test_df = pd.read_excel("/content/drive/MyDrive/dataset/test.xlsx")
test_df["audio_path"] = test_df["file_name"].apply(
    lambda x: f"/content/drive/MyDrive/dataset/test/{x.strip()}"
)
test_df['dialect'] = test_df['file_name'].apply(
    lambda fn: (re.search(r'test_([a-z]+)_', str(fn)).group(1) if re.search(r'test_([a-z]+)_', str(fn)) else None)
)

print(f"\nRunning zero-shot on {len(test_df)} test samples (~20 min)...")
raw_preds, refs, dialects, files = [], [], [], []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    try:
        audio, _ = librosa.load(row["audio_path"], sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = base_model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    except Exception as e:
        pred = ""
    raw_preds.append(pred)
    refs.append(str(row["transcripts"]))
    dialects.append(row["dialect"])
    files.append(row["file_name"])

# Save predictions
OUT_DIR = "/content/drive/MyDrive/claudefinetuneandklmn"
baseline_df = pd.DataFrame({
    "file_name": files, "dialect": dialects, "reference": refs, "prediction": raw_preds,
})
baseline_df.to_csv(f"{OUT_DIR}/baseline_zeroshot_predictions.csv", index=False)

# Compute WER using SAME method as fine-tune (pooled jiwer)
print("\n" + "=" * 55)
print("ZERO-SHOT INDICWAV2VEC — BASELINE (TEST SET)")
print("=" * 55)
print(f"{'Dialect':<14} {'N':>5} {'Greedy':>9}")
print("-" * 32)

per_dialect = {}
for d in sorted(set(dialects)):
    if d is None: continue
    sub = [(r, p) for r, p, dia in zip(refs, raw_preds, dialects) if dia == d]
    r_list = [x[0] for x in sub]
    p_list = [x[1] for x in sub]
    p_list_clean = [p if p.strip() else " " for p in p_list]  # avoid jiwer crash on empty
    wer = 100 * jiwer.wer(r_list, p_list_clean)
    per_dialect[d] = {"n": len(r_list), "wer": wer}

# Sort by WER ascending
for d in sorted(per_dialect.keys(), key=lambda x: per_dialect[x]['wer']):
    v = per_dialect[d]
    print(f"{d:<14} {v['n']:>5} {v['wer']:>8.1f}%")

macro = np.mean([v['wer'] for v in per_dialect.values()])
all_preds_clean = [p if p.strip() else " " for p in raw_preds]
micro = 100 * jiwer.wer(refs, all_preds_clean)

print("-" * 32)
print(f"{'MEAN WER':<14} {'':>5} {macro:>8.1f}%  (macro)")
print(f"{'MICRO':<14} {len(refs):>5} {micro:>8.1f}%  (overall)")
print("-" * 32)

# Save summary
import json
with open(f"{OUT_DIR}/baseline_zeroshot_results.json", "w") as f:
    json.dump({
        "per_dialect": per_dialect,
        "macro_wer": round(macro, 1),
        "micro_wer": round(micro, 1),
        "n_total": len(refs),
        "note": "Zero-shot IndicWav2Vec on test set. WER computed via pooled jiwer (same method as fine-tune).",
    }, f, indent=2)

print(f"\n✓ Predictions saved: baseline_zeroshot_predictions.csv")
print(f"✓ Summary saved: baseline_zeroshot_results.json")

# Free GPU memory
del base_model
torch.cuda.empty_cache()
