# Stage 2: Fine-tuning (greedy decoding)

Paper: Section 4.1, Table 3.

| Scoring | WER | CER | SD of WER |
|---|---|---|---|
| Raw | 82.5 | 42.1 | 12.7 |
| BnUnicode (final) | 77.4 | 39.9 | 14.7 |

WER and CER are means over the 12 dialects. SD is the population standard deviation of the 12 per-dialect WER values.

- `code/`: scripts that produce this stage
- `results/`: saved scores
- `predictions/`: saved test predictions (not saved for greedy; see below)
- `reproduce.py`: recomputes the row above from the saved files

Run: `pip install jiwer numpy pandas` and `pip install --no-deps bnunicodenormalizer`, then `python reproduce.py`.
