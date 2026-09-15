# Data Size Effects on Per-Dialect WER

## Key Finding

Training data size does NOT significantly predict per-dialect WER:
- Pearson r=0.169 (p=0.60), Spearman r=0.385 (p=0.22)

Comilla: 43.4% WER with only 254 training samples (best).
Chittagong: 87.4% WER with 1,405 samples (worst).
Sylhet: 74.9% WER with 6,100 samples (most data, middling result).

Conclusion: dialect difficulty is driven by linguistic divergence from
standard Bengali, not by data quantity. This motivates the linguistic analysis.

## Run

    pip install -r requirements.txt
    python datasize_analysis.py

(Place dataset's train.xlsx in this folder, or adjust the path.)

## Files

- datasize_analysis.py - the analysis code
- DATASIZE_REPORT.txt - formatted report
- datasize_vs_wer.csv - the table
- datasize_results.json - results
- requirements.txt

Generated: 2026-05-23 10:23

## Data Sources (for reviewers)

- train.xlsx: from the RegSpeech12 dataset folder (not duplicated here).
  Place it in this folder or adjust the path in datasize_analysis.py to run.
-  Per-dialect WER values: from the final BnUnicodeNormalizer-based evaluation
(72.2% mean) in the '5gm+normalization-72wer' folder.
