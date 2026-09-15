# 5-gram LM + Normalization (Best Result: 72.3% Mean WER)

Fine-tuned IndicWav2Vec + 5-gram KenLM + text normalization on RegSpeech12
(12 Bengali dialects, 2,132 test samples).

## Headline Result

5-gram KenLM (beam 200) + normalization = **72.3% Mean WER** (macro average of 12 per-dialect WERs)

## Matched-norm comparison

`fair_comparison_bnnorm.json` — the same 5-gram predictions re-scored with
BnUnicodeNormalizer (matching prior work's evaluation convention): 72.2% Mean WER,
41.3% Mean CER. This is the paper's reported final headline number.

## Verify fastest (~5 sec, no GPU)

    pip install "numpy<2" jiwer pandas
    python reproduce.py

Expected: MEAN WER 72.3%

## Re-decode from logits (~15 min, CPU)

    pip install -r requirements.txt
    pip install https://github.com/kpu/kenlm/archive/master.zip
    python 01_build_5gram_kenlm.py
    python 02_decode_5gram_and_normalize.py


## Settings

- Base: fine-tuned ai4bharat/indicwav2vec_v1_bengali (RegSpeech12 fine-tune)
- LM: 5-gram KenLM, modified Kneser-Ney (--discount_fallback), corpus = train+valid transcripts x5
- Decoding: alpha=0.5, beta=1.5, beam_width=200
- Normalization: NFC + remove punctuation + collapse whitespace (applied to refs AND preds)
- Processor loaded from ai4bharat/indicwav2vec_v1_bengali (NOT from model folder)

## Files

CODE:
- 01_build_5gram_kenlm.py - builds the 5-gram LM
- 02_decode_5gram_and_normalize.py - decode + normalize + WER
- reproduce.py - fast verify from saved predictions
- requirements.txt

DATA:
- lm_5gram.bin - trained 5-gram LM
- test_logits.pkl - fine-tuned model outputs on test set
- kenlm_training_corpus.txt - LM training text
- test_preds_5gram.pkl - saved 5-gram predictions
- test_predictions_5gram_full.csv - all predictions (raw + normalized)
- per_dialect_5gram.csv - per-dialect table
- results_5gram.json - summary
- fair_comparison_bnnorm.json - same predictions, BnUnicodeNormalizer-based scoring (72.2%/41.3%)


Generated: 2026-05-20 20:15
