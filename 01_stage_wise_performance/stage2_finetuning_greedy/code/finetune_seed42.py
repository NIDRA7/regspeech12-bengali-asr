"""
Fine-tune IndicWav2Vec (ai4bharat/indicwav2vec_v1_bengali) on RegSpeech12 train split, seed 42 (main run).
3 epochs, AdamW, lr 3e-5, 500 warm-up, batch 6 x accum 2, fp16, CNN feature encoder frozen.
Output: fine-tuned model folder (set OUT below).
"""

import os, numpy as np, pandas as pd, librosa, jiwer, torch
from dataclasses import dataclass
from typing import Any
from datasets import Dataset
from transformers import AutoProcessor, AutoModelForCTC, TrainingArguments, Trainer
from torch.nn.utils.rnn import pad_sequence

BASE = "/content/drive/MyDrive/dataset"
OUT = "/content/drive/MyDrive/indicwav2vec_regspeech12_ft"
MODEL = "ai4bharat/indicwav2vec_v1_bengali"

def load_split(split):
    df = pd.read_excel(f"{BASE}/{split}.xlsx")
    df["audio_path"] = df["file_name"].apply(lambda x: f"{BASE}/{split}/{x.strip()}")
    df = df.rename(columns={"transcripts": "sentence"}).dropna(subset=["audio_path", "sentence"])
    return df[df["audio_path"].apply(os.path.exists)]

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForCTC.from_pretrained(MODEL)
model.freeze_feature_encoder()

@dataclass
class Collator:
    processor: Any
    tokenizer: Any
    def __call__(self, features):
        ivs, labs = [], []
        for f in features:
            audio, _ = librosa.load(f["audio_path"], sr=16000)
            ivs.append({"input_values": audio})
            labs.append(self.tokenizer(f["sentence"], return_tensors="pt").input_ids[0])
        batch = self.processor.feature_extractor.pad(ivs, padding="longest", return_tensors="pt")
        batch["labels"] = pad_sequence(labs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return batch

def metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    return {"wer": 100 * jiwer.wer(processor.batch_decode(pred.label_ids, group_tokens=False), processor.batch_decode(pred_ids))}

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUT, remove_unused_columns=False, per_device_train_batch_size=4,
        gradient_checkpointing=True, fp16=True, num_train_epochs=12,
        eval_strategy="steps", save_steps=500, eval_steps=500, logging_steps=100,
        learning_rate=3e-5, warmup_steps=1000, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        report_to="none",
    ),
    train_dataset=Dataset.from_pandas(load_split("train")),
    eval_dataset=Dataset.from_pandas(load_split("valid")),
    data_collator=Collator(processor=processor, tokenizer=processor.tokenizer),
    compute_metrics=metrics,
)
trainer.train()
trainer.save_model(f"{OUT}/FINAL_MODEL")
processor.save_pretrained(f"{OUT}/FINAL_MODEL")
