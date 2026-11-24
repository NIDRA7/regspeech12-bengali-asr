# Stage 5: Fine-tuning + global 5-gram LM (final)

Paper: Section 4.1, Table 3.

| Scoring | WER | CER | SD of WER |
|---|---|---|---|
| Raw | 78.1 | 43.3 | 12.1 |
| BnUnicode (final) | 71.8 | 40.5 | 14.2 |

WER and CER are means over the 12 dialects. SD is the population standard deviation of the 12 per-dialect WER values.

- `code/`: scripts that produce this stage
- `results/`: saved scores and tuned decoding settings
- `predictions/`: saved test predictions
- `reproduce.py`: recomputes the row above from the saved files

Run: `pip install jiwer numpy pandas` and `pip install --no-deps bnunicodenormalizer`, then `python reproduce.py`.
