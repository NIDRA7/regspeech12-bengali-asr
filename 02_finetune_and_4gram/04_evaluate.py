"""
04_evaluate.py
Evaluates fine-tuned model on test set with greedy + global KenLM decoding.
Produces the 85.5% / 80.3% WER results.

Requirements:
    pip install transformers pyctcdecode jiwer pandas
    pip install https://github.com/kpu/kenlm/archive/master.zip
"""

import pickle, numpy as np, pandas as pd, re, jiwer, json, os
from transformers import AutoProcessor
from pyctcdecode import build_ctcdecoder

HERE = os.path.dirname(os.path.abspath(__file__))

# Load saved logits and processor
with open(f"{HERE}/test_logits.pkl", "rb") as f:
    td = pickle.load(f)

processor = AutoProcessor.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")

vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
vocab_list = [t if t != processor.tokenizer.word_delimiter_token else " " for t, _ in sorted_vocab]

# Build LM decoder
decoder = build_ctcdecoder(
    labels=vocab_list,
    kenlm_model_path=f"{HERE}/lm_4gram.bin",
    alpha=0.5, beta=1.5,
)

# Decode all samples
greedy_preds, lm_preds = [], []
for logits in td["logits"]:
    lf32 = logits.astype(np.float32)
    greedy_preds.append(processor.batch_decode(np.argmax(lf32, axis=-1)[None], skip_special_tokens=True)[0])
    lm_preds.append(decoder.decode(lf32, beam_width=100))

# Tag dialects
test_df = pd.read_excel("/content/drive/MyDrive/dataset/test.xlsx")
test_df["dialect"] = test_df["file_name"].apply(
    lambda fn: (re.search(r"test_([a-z]+)_", str(fn)).group(1)
                if re.search(r"test_([a-z]+)_", str(fn)) else None)
)

# Per-dialect WER
per_dialect = {}
for d in sorted(test_df["dialect"].dropna().unique()):
    mask = (test_df["dialect"] == d).tolist()
    refs = [r for r, m in zip(td["refs"], mask) if m]
    g    = [p for p, m in zip(greedy_preds, mask) if m]
    l    = [p if p.strip() else " " for p, m in zip(lm_preds, mask) if m]
    per_dialect[d] = {
        "n": len(refs),
        "greedy": round(100 * jiwer.wer(refs, g), 1),
        "lm":     round(100 * jiwer.wer(refs, l), 1),
    }

mean_greedy = round(np.mean([v["greedy"] for v in per_dialect.values()]), 1)
mean_lm     = round(np.mean([v["lm"]     for v in per_dialect.values()]), 1)

print(f"{'Dialect':<14} {'N':>5} {'Greedy':>10} {'+KenLM':>10}")
print("-" * 45)
for d, v in sorted(per_dialect.items(), key=lambda x: x[1]["greedy"]):
    print(f"{d:<14} {v['n']:>5} {v['greedy']:>9.1f}% {v['lm']:>9.1f}%")
print("-" * 45)
print(f"{'MEAN WER':<14} {'':>5} {mean_greedy:>9.1f}% {mean_lm:>9.1f}%")
print(f"(Expected: 82.5% / 78.8%)")
