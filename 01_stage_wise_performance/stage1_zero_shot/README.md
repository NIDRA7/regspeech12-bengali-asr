# Stage 1: Zero-shot IndicWav2Vec

Paper: Section 4.1, Table 3.

| Scoring | WER | CER | SD of WER |
|---|---|---|---|
| Raw | 89.0 | 60.1 | 11.3 |
| BnUnicode (final) | 86.6 | 58.7 | 13.3 |

WER and CER are means over the 12 dialects. SD is the population standard deviation of the 12 per-dialect WER values.

- `code/`: scripts that produce this stage
- `results/`: saved scores
- `predictions/`: saved test predictions
- `reproduce.py`: recomputes the row above from the saved files

Run: `pip install jiwer numpy pandas` and `pip install --no-deps bnunicodenormalizer`, then `python reproduce.py`.
