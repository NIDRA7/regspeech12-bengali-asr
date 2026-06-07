"""
01_finetune.py
Fine-tune IndicWav2Vec on RegSpeech12 → produces 85.5% greedy WER on test set.

Requirements:
    pip install transformers librosa jiwer pandas tqdm openpyxl accelerate datasets

Run on T4 GPU (~4-6 hours):
    python 01_finetune.py
"""

import os, numpy as np, pandas as pd, librosa, jiwer, torch
from dataclasses import dataclass
from typing import Any
from datasets import Dataset
from transformers import AutoProcessor, AutoModelForCTC, TrainingArguments, Trainer
from torch.nn.utils.rnn import pad_sequence

BASE_PATH = "/content/drive/MyDrive/dataset"
OUTPUT_DIR = "/content/drive/MyDrive/indicwav2vec_regspeech12_ft"
MODEL_NAME = "ai4bharat/indicwav2vec_v1_bengali"
SAMPLING_RATE = 16000

def load_split(split):
    df = pd.read_excel(f"{BASE_PATH}/{split}.xlsx")
    df["audio_path"] = df["file_name"].apply(lambda x: f"{BASE_PATH}/{split}/{x.strip()}")
    df = df.rename(columns={"transcripts": "sentence"}).dropna(subset=["audio_path", "sentence"])
    return df[df["audio_path"].apply(os.path.exists)]

train_df = load_split("train")
valid_df = load_split("valid")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForCTC.from_pretrained(MODEL_NAME)
model.freeze_feature_encoder()

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

@dataclass
class DataCollatorCTC:
    processor: Any
    tokenizer: Any
    def __call__(self, features):
        ivs, labs = [], []
        for f in features:
            audio, _ = librosa.load(f["audio_path"], sr=SAMPLING_RATE)
            ivs.append({"input_values": audio})
            labs.append(self.tokenizer(f["sentence"], return_tensors="pt").input_ids[0])
        batch = self.processor.feature_extractor.pad(ivs, padding="longest", return_tensors="pt")
        batch["labels"] = pad_sequence(labs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return batch

collator = DataCollatorCTC(processor=processor, tokenizer=processor.tokenizer)

def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    return {"wer": 100 * jiwer.wer(label_str, pred_str)}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    gradient_checkpointing=True,
    fp16=True,
    num_train_epochs=12,
    eval_strategy="steps",
    save_steps=500, eval_steps=500, logging_steps=100,
    learning_rate=3e-5, warmup_steps=1000,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="wer", greater_is_better=False,
    report_to="none",
)

trainer = Trainer(model=model, args=training_args,
                  train_dataset=train_dataset, eval_dataset=valid_dataset,
                  data_collator=collator, compute_metrics=compute_metrics)
trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/FINAL_MODEL")
processor.save_pretrained(f"{OUTPUT_DIR}/FINAL_MODEL")
