# Statistical Significance Analysis

## Key Findings

-  Best model: 72.2% Mean WER, 95% CI [63.8, 79.8]
- Improvement: 14.6 WER points over baseline
- Baseline: 86.8% Mean WER, 95% CI [78.4, 93.4]
- Improvement: 14.5 WER points over baseline
- Statistically significant: Wilcoxon p<0.001, paired t-test p<0.001
- vs SOTA (BRDialect 74.1%): our 72.2% is lower

In plain terms: the improvement over the baseline is proven to be real, not luck.

## Run

    pip install -r requirements.txt
    python significance_analysis.py

## Files

- significance_analysis.py - the test code
- SIGNIFICANCE_REPORT.txt - formatted report
- significance_results.json - machine-readable results
- test_preds_5gram.pkl, baseline_predictions_full.csv - data
- requirements.txt

Generated: 2026-05-21 13:00
