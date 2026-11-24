# Stage 4: Fine-tuning + dialect-aware 4-gram LMs (tested)

Paper: Section 4.1, Table 3.

| Scoring | WER | CER | SD of WER |
|---|---|---|---|
| Raw | 78.8 | 43.4 | 10.8 |
| BnUnicode (final) | 72.7 | 40.6 | 12.6 |

WER and CER are means over the 12 dialects. SD is the population standard deviation of the 12 per-dialect WER values.

- `code/`: scripts that produce this stage
- `results/`: saved scores and tuned decoding settings
- `predictions/`: saved test predictions
- `reproduce.py`: recomputes the row above from the saved files

Run: `pip install jiwer numpy pandas` and `pip install --no-deps bnunicodenormalizer`, then `python reproduce.py`.
