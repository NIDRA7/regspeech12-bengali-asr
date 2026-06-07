Global vs Dialect-Aware KenLM on RegSpeech12
=============================================

Compares global 4-gram KenLM with per-dialect KenLMs, with
per-dialect alpha/beta tuning on validation, applied to test set.

RESULTS (test set, 2132 samples):
  Fine-tuned (greedy)                    Mean WER 85.5%
  + Global KenLM (default a/b)           Mean WER 80.3%
  + Tuned a/b + Dialect-aware LM         Mean WER ~79.5%

PIPELINE (which code produces what):

  Step 5 -> 05_build_dialect_kenlms.py
            Builds one KenLM per dialect (10x upweighted).
            Output: dialect_lms/lm_<dialect>.bin (12 files)

  Step 6 -> 06_grid_search_valid.py
            Grid search on VALID set for each dialect.
            Finds best LM (global vs dialect) and best alpha/beta.
            Output: best_lm_settings.json

  Step 7 -> 07_evaluate_dialect_aware.py  <-- MAIN RESULT
            Applies the winning settings to TEST set.
            Output: test_results_final.json + per-dialect WER table

HOW TO REPRODUCE:

Quick (use saved best_lm_settings.json, skip Step 5 and 6, ~15-20 min CPU):
  pip install -r requirements.txt
  pip install https://github.com/kpu/kenlm/archive/master.zip
  python 07_evaluate_dialect_aware.py

Full pipeline from scratch (~1 hour CPU):
  python 05_build_dialect_kenlms.py     # 15 min - builds 12 LMs
  python 06_grid_search_valid.py        # 30 min - tunes alpha/beta
  python 07_evaluate_dialect_aware.py   # 15-20 min - final eval

FILES:

CODE:
  05_build_dialect_kenlms.py    builds 12 dialect KenLMs
  06_grid_search_valid.py       finds best alpha/beta per dialect
  07_evaluate_dialect_aware.py  produces final test WER
  requirements.txt              Python deps

DATA (model outputs - reused from greedy+LM step):
  test_logits.pkl               fine-tuned model outputs on test
  valid_logits.pkl              fine-tuned model outputs on valid
  lm_4gram.bin                  global KenLM
  dialect_lms/lm_<d>.bin        per-dialect KenLMs (12 files)

RESULTS:
  best_lm_settings.json         per-dialect best alpha/beta + winner
  test_predictions_full.csv     all test predictions
  test_results_final.json       summary WER numbers
  test_results_table.*          paper table (txt/csv/tex)

HYPERPARAMETERS:
  Dialect KenLM: 4-gram (3-gram if <5000 corpus lines),
                 transcripts 10x upweighted in-domain
  Grid search:   alpha in {0.3, 0.5, 0.8, 1.2}
                 beta  in {0.0, 1.0, 2.0}
                 beam_width = 50 (valid), 100 (test)
