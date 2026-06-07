"""
06_grid_search_valid.py
Per-dialect grid search on VALID set: find best LM (global vs dialect) and best alpha/beta.
Produces: best_lm_settings.json (used by 07_evaluate_dialect_aware.py).

Needs in same folder:
  valid_logits.pkl, lm_4gram.bin, dialect_lms/lm_<dialect>.bin

Run: CPU, ~30-40 min.
Setup:
  pip install transformers pyctcdecode jiwer pandas openpyxl numpy<2
  pip install https://github.com/kpu/kenlm/archive/master.zip
"""
import pickle, pandas as pd, re, numpy as np, jiwer, json, os, warnings, logging
from pyctcdecode import build_ctcdecoder
from transformers import AutoProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
GLOBAL_LM = f"{HERE}/lm_4gram.bin"
DIALECT_LM_DIR = f"{HERE}/dialect_lms"

processor = AutoProcessor.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")
vocab = processor.tokenizer.get_vocab()
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
vocab_list = [t if t != processor.tokenizer.word_delimiter_token else " " for t, _ in sorted_vocab]

with open(f"{HERE}/valid_logits.pkl", "rb") as f:
    vd = pickle.load(f)

valid_df = pd.read_excel("/content/drive/MyDrive/dataset/valid.xlsx")
valid_df["dialect"] = valid_df["file_name"].apply(
    lambda fn: (re.search(r"valid_([a-z]+)_", str(fn)).group(1)
                if re.search(r"valid_([a-z]+)_", str(fn)) else None)
)

valid_by_dialect = {}
for d in valid_df["dialect"].dropna().unique():
    idx = valid_df.index[valid_df["dialect"] == d].tolist()
    valid_by_dialect[d] = {
        "logits": [vd["logits"][i] for i in idx],
        "refs":   [vd["refs"][i]   for i in idx],
    }

ALPHAS = [0.3, 0.5, 0.8, 1.2]
BETAS = [0.0, 1.0, 2.0]

def grid_search(logits_list, refs, lm_path):
    best = {"alpha": None, "beta": None, "wer": 100.0}
    for alpha in ALPHAS:
        for beta in BETAS:
            try:
                d = build_ctcdecoder(labels=vocab_list, kenlm_model_path=lm_path, alpha=alpha, beta=beta)
                preds = [d.decode(lg.astype(np.float32), beam_width=50) for lg in logits_list]
                w = 100 * jiwer.wer(refs, preds)
                if w < best["wer"]:
                    best = {"alpha": alpha, "beta": beta, "wer": w}
            except Exception:
                pass
    return best

warnings.filterwarnings("ignore")
logging.getLogger("pyctcdecode").setLevel(logging.ERROR)

best_settings = {}
print(f"{'Dialect':<14} {'GlobalLM':>10} {'DialectLM':>10} {'Winner':>10}")
print("-" * 50)
for dialect in sorted(valid_by_dialect.keys()):
    vlogits = valid_by_dialect[dialect]["logits"]
    vrefs   = valid_by_dialect[dialect]["refs"]
    g = grid_search(vlogits, vrefs, GLOBAL_LM)
    d = grid_search(vlogits, vrefs, f"{DIALECT_LM_DIR}/lm_{dialect}.bin")
    winner = "dialect" if d["wer"] < g["wer"] else "global"
    best_settings[dialect] = {"global": g, "dialect": d, "winner": winner}
    print(f"{dialect:<14} {g['wer']:>9.1f}% {d['wer']:>9.1f}% {winner:>10}")

with open(f"{HERE}/best_lm_settings.json", "w") as f:
    json.dump(best_settings, f, indent=2)
print(f"\nSaved: {HERE}/best_lm_settings.json")
