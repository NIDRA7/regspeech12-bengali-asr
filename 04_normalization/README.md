# Text Normalization for RegSpeech12 Results

## What this does

Applies safe text normalization (NFC + remove punctuation + collapse whitespace)
to predictions + references for all 4 evaluation stages, then recomputes Mean WER.

## Headline result

**Mean WER drops from 78.6% to 73.1% after normalization.**
5.5 percentage points free improvement, no model changes.

## Full results

| Method                  | Before  | After   | Drop  |
|-------------------------|---------|---------|-------|
| Zero-shot baseline      | 89.0%   | 86.8%   | -2.2 |
| Fine-tune (greedy)      | 82.5%   | 77.7%   | -4.8 |
| + Global KenLM          | 78.8%   | 72.9%   | -5.9 |
| + Dialect-aware LM      | 78.6%   | 73.1%   | -5.5 |

## Normalization steps

1. Unicode NFC normalization (canonical form for Bengali characters)
2. Remove punctuation (Bengali + English)
3. Collapse multiple whitespace into single space
4. Strip leading/trailing whitespace

This is safe normalization - standard practice in ASR papers.
Does NOT alter Bengali words, only removes spurious mismatches.

## Files

- apply_normalization.py - the code
- normalized_wer.json - machine-readable summary
- normalized_table.txt - human-readable table
- normalized_results.csv - overall CSV
- normalized_per_dialect.csv - per-dialect CSV
- normalized_table.tex - LaTeX for paper
- requirements.txt - dependencies
- README.md - this file

## How to reproduce

    pip install -r requirements.txt
    python apply_normalization.py

Generated: 2026-05-18 07:26
