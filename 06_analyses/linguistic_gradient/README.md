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

## IMPORTANT - Citation Required

The dialect groupings are based on Bengali dialectology and MUST be cited to an
authoritative source (e.g., Grierson's Linguistic Survey of India, or a modern
Bengali dialectology reference) before publication. Verify group membership against
that source and adjust if needed. The WER/CER numbers are from this project's
best result; the groupings are interpretive and need a citation.

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
