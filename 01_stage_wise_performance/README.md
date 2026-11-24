# 4.1 Stage-wise performance (paper Table 3)

WER, CER (means over the 12 dialects) and SD of WER (population standard deviation across dialects), in %.

| Stage | System | Raw WER | Raw CER | Raw SD | BnUnicode WER | BnUnicode CER | BnUnicode SD |
|---|---|---|---|---|---|---|---|
| 1 | [Zero-shot IndicWav2Vec](stage1_zero_shot) | 89.0 | 60.1 | 11.3 | 86.6 | 58.7 | 13.3 |
| 2 | [Fine-tuning (greedy)](stage2_finetuning_greedy) | 82.5 | 42.1 | 12.7 | 77.4 | 39.9 | 14.7 |
| 3 | [Fine-tuning + global 4-gram LM (tested)](stage3_global_4gram) | 78.2 | 43.1 | 12.1 | 71.9 | 40.3 | 14.2 |
| 4 | [Fine-tuning + dialect-aware 4-gram LMs (tested)](stage4_dialect_aware_4gram) | 78.8 | 43.4 | 10.8 | 72.7 | 40.6 | 12.6 |
| 5 | [Fine-tuning + global 5-gram LM (final)](stage5_global_5gram_final) | 78.1 | 43.3 | 12.1 | 71.8 | 40.5 | 14.2 |

Stages 3 to 5 are three language-model configurations tested separately on the fine-tuned model, not sequential steps.

Each stage folder has `code/`, `results/`, `predictions/` and a `reproduce.py` that recomputes its row of the table.
The scoring functions shared by all folders are in `../common/scoring.py`.
