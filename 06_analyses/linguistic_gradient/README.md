# Linguistic Difficulty Analysis

## Key Finding

WER rises monotonically with linguistic distance from Standard Bengali:

| Group | Mean WER | Mean CER |
|---|---|---|
| Rarhi / Central (closest) | 57.8% | 28.1% |
| Bangali / East-Central | 63.6% | 32.8% |
| Sylheti | 74.9% | 43.7% |
| Kamta / Rangpuri | 80.8% | 48.6% |
| Chittagonian (most divergent) | 84.9% | 54.2% |
| kishoreganj (uncertain) | 87.3% | 59.2% |

This linguistic gradient - NOT data size - explains dialect difficulty.

## Dialect Groupings

The dialect groupings follow Hassan et al. 2025 (RegSpeech12) and Masica 1991
(The Indo-Aryan Languages), used here as an exploratory qualitative grouping
(see paper Section 4.8) rather than a formal linguistic-distance measure.
WER/CER values are from the final BnUnicodeNormalizer-based setting (72.2% mean).
## Run

    pip install -r requirements.txt
    python linguistic_analysis.py

## Files

- linguistic_analysis.py - the analysis code
- LINGUISTIC_REPORT.txt - formatted report
- linguistic_groups_wer.csv - the grouping table
- linguistic_results.json - results
- requirements.txt

Generated: 2026-05-23 10:46
