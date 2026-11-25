# 4.2.1 Normalization (paper Table 4)

Effect of the three scoring settings on WER and CER (means over the 12 dialects, %).

| System | Raw WER | Raw CER | Standard WER | Standard CER | BnUnicode WER | BnUnicode CER |
|---|---|---|---|---|---|---|
| Zero-shot IndicWav2Vec | 89.0 | 60.1 | 86.8 | 58.4 | 86.6 | 58.7 |
| Fine-tuned, greedy | 82.5 | 42.1 | 77.7 | 40.2 | 77.4 | 39.9 |
| + global 4-gram LM | 78.2 | 43.1 | 72.0 | 40.9 | 71.9 | 40.3 |
| + dialect-aware 4-gram LMs | 78.8 | 43.4 | 73.0 | 41.3 | 72.7 | 40.6 |
| + global 5-gram LM | 78.1 | 43.3 | 71.9 | 41.1 | 71.8 | 40.5 |

- Raw: no normalization. Standard: Unicode NFC + punctuation removal + whitespace normalization. BnUnicode: BnUnicodeNormalizer + punctuation removal.
- `code/score_all_normalizations.py`: the notebook script that scored every stage under the three settings.
- `results/final_scores_all_norms.json`: saved per-dialect WER/CER for each stage and scoring.
- `reproduce.py`: recomputes the table from the saved predictions in `01_stage_wise_performance`. Greedy predictions are not stored (they need the logits file), so the greedy row is checked from the stored scores.

Run: `pip install jiwer numpy pandas` and `pip install --no-deps bnunicodenormalizer`, then `python reproduce.py`.
