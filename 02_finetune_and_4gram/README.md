# Fine-Tuned IndicWav2Vec + 4-gram KenLM

Stage 2 of the RegSpeech12 Bengali dialect ASR pipeline. Fine-tunes IndicWav2Vec on the
RegSpeech12 training set and rescores predictions with a global 4-gram KenLM.

## Headline numbers

| Configuration             | Mean WER |
|---------------------------|----------|
| Fine-tune (greedy)        | **82.5%** |
| + Global 4-gram KenLM     | **78.8%** |
| Drop from LM rescoring    | -3.7     |

Mean WER is the average of per-dialect WERs across the 12 RegSpeech12 dialects (test set,
2,132 utterances).

## How to reproduce

### Fastest verification (~10 seconds, no GPU)

Recomputes Mean WER directly from the saved predictions CSV:

```bash
pip install jiwer pandas numpy
python reproduce.py
```

Expected output: Greedy 82.5%, +KenLM 78.8%.

### Full pipeline from scratch (T4 GPU, ~5-6 hours)

```bash
pip install -r requirements.txt
pip install https://github.com/kpu/kenlm/archive/master.zip
python 01_finetune.py         # fine-tune the acoustic model (4-6 hours, T4 GPU)
python 02_compute_logits.py   # save logits for fast LM re-decoding (20-30 min)
python 03_build_kenlm.py      # build the 4-gram KenLM (~10 min)
python 04_evaluate.py         # evaluate greedy + LM (~5 min)
```

Note: the trained model weights (1.2 GB) and saved logits (302 MB) are not included
in this repository due to size. Download from [HuggingFace Hub or Drive link to add].

## Settings

| Setting             | Value |
|---------------------|-------|
| Base model          | ai4bharat/indicwav2vec_v1_bengali |
| Train / Valid split | 17,049 / 2,132 utterances |
| Optimizer           | AdamW (fused) |
| Learning rate       | 3e-5 (linear schedule, 500 warm-up steps) |
| Epochs              | 3 |
| Batch size          | 6 per device, gradient accumulation 2 (effective 12) |
| Mixed precision     | fp16 |
| Random seed         | 42 |
| Feature encoder     | frozen |
| Audio               | 16 kHz mono |
| Output vocabulary   | 87 tokens (Bengali character-level) |
| KenLM order         | 4-gram, modified Kneser-Ney |
| LM training corpus  | in-domain RegSpeech12 transcripts (upweighted by repetition), 95,845 lines |
| LM alpha (weight)   | 0.5 |
| LM beta (insertion) | 1.5 |
| Beam width          | 100 |

## Files in this folder

| File | Description |
|------|-------------|
| `01_finetune.py`              | Fine-tuning script |
| `02_compute_logits.py`        | Logits computation for fast LM re-decoding |
| `03_build_kenlm.py`           | KenLM 4-gram build script |
| `04_evaluate.py`              | Greedy + LM evaluation |
| `reproduce.py`                | Fast WER verification from saved CSV |
| `results.json`                | Mean WER and per-dialect numbers (machine-readable) |
| `results_table.txt`           | Per-dialect results (human-readable) |
| `results_table.csv`           | Per-dialect results (CSV) |
| `test_predictions_full.csv`   | All 2,132 test predictions (greedy + KenLM) |
| `requirements.txt`            | Python dependencies |
| `README.md`                   | This file |
