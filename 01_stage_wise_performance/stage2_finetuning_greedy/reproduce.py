"""Reproduce the Stage 2: Fine-tuning (greedy decoding) row of Table 3 (Raw and BnUnicode: WER, CER, SD).
Usage: python reproduce.py"""
import sys, os, glob, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from scoring import score_by_dialect, check
HERE = os.path.dirname(os.path.abspath(__file__))
# Greedy predictions are not stored (they come from the 300 MB logits file). Scores are stored instead.
S = json.load(open(os.path.join(HERE, "results", "stage2_finetuning_greedy_scores.json")))
import numpy as np
ok = True
for name, key, exp in (("raw", "none", (82.5, 42.1, 12.7)), ("bnunicode", "bnunicode", (77.4, 39.9, 14.7))):
    w = np.array(list(S[key]["wer_per_dialect"].values())); c = np.array(list(S[key]["cer_per_dialect"].values()))
    ok &= check("Stage 2 " + name, (w.mean(), c.mean(), w.std(ddof=0)), exp)
print("To regenerate from logits: run code/compute_logits_seed42.py, then greedy-decode the logits (see README).")
sys.exit(0 if ok else 1)
