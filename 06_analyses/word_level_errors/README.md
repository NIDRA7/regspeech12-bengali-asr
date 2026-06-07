# Word-Level & Bengali-Specific Error Analysis

## Key Findings

1. Errors are 61.3% substitutions, 32.9% deletions, 5.9% insertions.
2. Per-dialect WER is strongly driven by DELETIONS (r=0.917, p<0.0001) and
   substitutions (r=0.855, p=0.0004) — harder dialects fail by omitting words.
3. The model under-produces complex Bengali characters: conjuncts (0.616) and
   independent vowels (0.608) are dropped ~40%, while simple consonants (0.847)
   and vowel signs (0.873) are better retained.

## Run

    pip install -r requirements.txt
    python word_level_error_analysis.py

## Files

- word_level_error_analysis.py - the analysis code
- ERROR_ANALYSIS_REPORT.txt - formatted report (both sections)
- error_breakdown_per_dialect.csv - per-dialect S/I/D table
- error_analysis.json - machine-readable results
- test_preds_5gram.pkl - predictions used
- requirements.txt

Generated: 2026-05-21 11:45
