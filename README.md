# RegSpeech12: Dialect-Aware ASR for 12 Bengali Regional Varieties

This repository contains the code, results, and analyses for a deep-learning automatic
speech recognition (ASR) system covering twelve Bengali regional dialects.

## Headline results

| System                                | Mean WER | Mean CER |
|---------------------------------------|----------|----------|
| Zero-shot baseline (IndicWav2Vec)     | 89.0%    | 58.4%    |
| + Fine-tuning (greedy decoding)       | 82.5%    | -        |
| + Global 4-gram KenLM                 | 78.8%    | -        |
| + Text normalization                  | 73.1%    | -        |
| + 5-gram KenLM + normalization        | 72.3%    | 41.8%    |
| Matched-norm (BnUnicodeNormalizer)    | 72.2%    | 41.3%    |

Total improvement over zero-shot baseline: 16.8 WER points.
All values are mean WER across the 12 RegSpeech12 dialects (test set, 2,132 utterances).

## Dataset

This work uses the RegSpeech12 corpus (Hassan et al., arXiv:2510.24096),
covering 12 Bengali regional dialects: Barishal, Chittagong, Comilla, Habiganj,
Kishoreganj, Narail, Narsingdi, Noakhali, Rangpur, Sandwip, Sylhet, Tangail.
Splits: 17,049 train / 2,132 validation / 2,132 test utterances.

The audio is not redistributed in this repository.
Obtain RegSpeech12 from the original source.

**See `00_dataset/README.md` for the full author list, license details (CC BY 4.0), citation, and setup instructions.**

## Repository structure

- 00_dataset/              Dataset documentation (RegSpeech12 source, license, splits, full citation)
- 01_baseline/              Zero-shot IndicWav2Vec baseline (89.0%)
- 02_finetune_and_4gram/    Fine-tuning + global 4-gram KenLM (82.5% -> 78.8%)
- 03_dialect_aware_lm/      12 per-dialect KenLMs comparison (78.8% vs 78.6%)
- 04_normalization/         Text normalization effect at each stage (-> 73.1%)
- 05_5gram_and_final/       Final system: 5-gram KenLM + normalization (72.3% / 72.2%)
- 06_analyses/              Supporting analyses (word-level, significance, data-size, linguistic, qualitative)

Each folder has its own README, source scripts, saved JSON/CSV results,
and a reproduce.py for fast verification of the headline numbers.

## Quick start

    cd 05_5gram_and_final/
    pip install -r requirements.txt
    python reproduce.py

Expected: Mean WER 72.3%, Mean CER 41.8%.

## Large files (not in this repo)

- Fine-tuned model weights (1.2 GB): HuggingFace Hub link to be added
- Saved logits (302 MB each): link to be added
- KenLM binaries (43-59 MB): link to be added
- LM corpus (49.7 MB): link to be added
- Audio: RegSpeech12 (arXiv:2510.24096)

## Methodology summary

- Base model: ai4bharat/indicwav2vec_v1_bengali
- Fine-tuning: 3 epochs, AdamW fused, LR 3e-5, linear schedule,
  500 warm-up, batch 6, grad accum 2 (effective 12), fp16,
  seed 42, feature encoder frozen
- LM: KenLM 4-gram and 5-gram, modified Kneser-Ney with --discount_fallback
- Decoding: pyctcdecode beam search, alpha=0.5, beta=1.5,
  beam 100 (4-gram) / 200 (5-gram)
- Normalization: Unicode NFC + remove punctuation + collapse whitespace (symmetric)
- Audio: 16 kHz mono, 87 Bengali character-level tokens

## Citation

Citation to be added once paper is published.

## License

MIT License. See LICENSE file.