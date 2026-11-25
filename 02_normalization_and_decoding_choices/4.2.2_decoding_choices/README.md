# 4.2.2 Decoding choices (paper Table 5)

Factorial ablation of the language model, its order, dialect-awareness and beam width, under three scorings (macro WER / CER, %).

| Language model | Order | Dialect-aware | Beam | Raw WER/CER | Standard WER/CER | BnUnicode WER/CER |
|---|---|---|---|---|---|---|
| None (greedy) | - | - | - | 82.5 / 42.1 | 77.7 / 40.2 | 77.4 / 39.9 |
| Global | 4 | No | 100 | 78.3 / 43.9 | 72.3 / 41.7 | 72.2 / 41.2 |
| Global | 4 | No | 200 | 78.2 / 43.1 | 72.0 / 40.9 | 71.9 / 40.3 |
| Dialect-aware | 4 | Yes | 100 | 78.8 / 43.8 | 73.1 / 41.6 | 72.8 / 41.0 |
| Dialect-aware | 4 | Yes | 200 | 78.7 / 43.1 | 72.9 / 40.9 | 72.6 / 40.3 |
| Global | 5 | No | 100 | 78.4 / 44.2 | 72.4 / 41.9 | 72.3 / 41.4 |
| Global | 5 | No | 200 | 78.1 / 43.3 | 71.9 / 41.1 | 71.8 / 40.5 |

- `code/decode_beam_variants.py`: decodes the test set at beam 100 (all LMs) and the dialect-aware LMs at beam 200.
- `code/factorial_ablation.py`: scores all configurations and writes the full table.
- `predictions/beam100/` and `predictions/dialect_beam200/`: saved test predictions for these runs. The beam-200 predictions of the global LMs are in `01_stage_wise_performance`.
- `results/table5_recomputed.json`: written by `reproduce.py`.
- `reproduce.py`: recomputes every row from the saved predictions.

Run: `pip install jiwer numpy pandas` and `pip install --no-deps bnunicodenormalizer`, then `python reproduce.py`.
