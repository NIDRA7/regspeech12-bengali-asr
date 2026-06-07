# Dataset — RegSpeech12 Bengali Regional Speech Corpus

This folder describes the dataset used in this work.
The dataset itself is NOT included in this repository.
Reviewers and users should obtain it from the original source.

---

## Source

**Dataset:** RegSpeech12 — A Regional Corpus of Bengali Spontaneous Speech Across Dialects

**arXiv link:** https://arxiv.org/abs/2510.24096

**DOI:** 10.48550/arXiv.2510.24096

**Submitted:** 28 October 2025

**License:** CC BY 4.0 (Creative Commons Attribution 4.0 International)
https://creativecommons.org/licenses/by/4.0/

The dataset was released for public use by its authors under the
CC BY 4.0 license, which allows redistribution and modification with
proper attribution to the original authors.

---

## Dataset coverage

RegSpeech12 contains approximately 100 hours of spontaneous speech
from twelve regional dialects of Bangladesh.

**Dialects covered (12):**
Barishal, Chittagong, Comilla, Habiganj, Kishoreganj, Narail,
Narsingdi, Noakhali, Rangpur, Sandwip, Sylhet, Tangail.

---

## Splits used in this study

| Split       | Utterances |
|-------------|------------|
| Train       | 17,049     |
| Validation  | 2,132      |
| Test        | 2,132      |
| **Total**   | **21,313** |

Per-dialect counts in the test set are documented in each result folder
of this repository (folders 01-05).

---

## How the dataset is used by the scripts

The scripts in folders 01-05 expect the dataset to follow this layout:

```
dataset/
  train.xlsx          # 17,049 rows
  valid.xlsx          # 2,132 rows
  test.xlsx           # 2,132 rows
  train/              # .wav audio files for the train split
  valid/              # .wav audio files for the validation split
  test/               # .wav audio files for the test split
```

Each xlsx file contains two columns:
- file_name    — name of the audio file (e.g., test_sylhet_001.wav)
- transcripts  — Bengali reference transcript for the utterance

The dialect label is encoded in the file_name itself, following the
pattern <split>_<dialect>_<index>.wav. The scripts extract the dialect
from the filename via regex.

**Audio format expected:**
- WAV format, 16 kHz mono, single-channel

---

## How to set up the dataset locally

Step 1. Obtain RegSpeech12 from the original source (link above).

Step 2. Place the dataset on your machine. The original experiments used:

    /content/drive/MyDrive/dataset/

(this was the path on Google Colab during the experiments).

Step 3. If your dataset is at a different path, edit the DATA_DIR variable
at the top of each script in folders 01-05.

---

## Why the dataset is not in this repository

- The audio files are multi-gigabyte and exceed GitHub size limits.
- The dataset is publicly available from its original authors under
  the CC BY 4.0 license.
- Linking to the original source ensures users get the authoritative
  version with any corrections or updates.
- Reviewers can verify the predictions saved in each folder
  (the *_predictions_full.csv files) against the original dataset.

---

## Verifying our results against the dataset

To verify the reported numbers without retraining:

1. Download RegSpeech12 (see arXiv link above)
2. Open any prediction CSV in this repository, for example:
   05_5gram_and_final/per_dialect_wer_cer.csv
3. Match the file names in the CSV to the downloaded test set
4. Run reproduce.py in each folder to recompute Mean WER

Expected headline numbers (from the results in this repository):
- Zero-shot baseline: 89.0% WER
- Final system (5-gram + normalization): 72.3% WER, 41.8% CER
- Matched-norm (BnUnicodeNormalizer): 72.2% WER, 41.3% CER

---

## Citation

If you use the RegSpeech12 dataset, please cite the original authors:

**Plain text:**

Md. Rezuwan Hassan, Azmol Hossain, Kanij Fatema, Rubayet Sabbir Faruque,
Tanmoy Shome, Ruwad Naswan, Trina Chakraborty, Md. Foriduzzaman Zihad,
Tawsif Tashwar Dipto, Nazia Tasnim, Nazmuddoha Ansary,
Md. Mehedi Hasan Shawon, Ahmed Imtiaz Humayun, Md. Golam Rabiul Alam,
Farig Sadeque, and Asif Sushmit. (2025). RegSpeech12: A Regional Corpus
of Bengali Spontaneous Speech Across Dialects. arXiv:2510.24096.
https://doi.org/10.48550/arXiv.2510.24096

**BibTeX:**

```
@article{hassan2025regspeech12,
  title   = {RegSpeech12: A Regional Corpus of Bengali Spontaneous
             Speech Across Dialects},
  author  = {{Md. Rezuwan Hassan} and {Azmol Hossain} and {Kanij Fatema}
             and {Rubayet Sabbir Faruque} and {Tanmoy Shome}
             and {Ruwad Naswan} and {Trina Chakraborty}
             and {Md. Foriduzzaman Zihad} and {Tawsif Tashwar Dipto}
             and {Nazia Tasnim} and {Nazmuddoha Ansary}
             and {Md. Mehedi Hasan Shawon} and {Ahmed Imtiaz Humayun}
             and {Md. Golam Rabiul Alam} and {Farig Sadeque}
             and {Asif Sushmit}},
  journal = {arXiv preprint},
  volume  = {arXiv:2510.24096},
  year    = {2025},
  doi     = {10.48550/arXiv.2510.24096},
  url     = {https://arxiv.org/abs/2510.24096}
}
```