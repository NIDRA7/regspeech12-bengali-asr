"""
07_evaluate_dialect_aware.py
Apply per-dialect winning LM + alpha/beta to TEST set.
Produces final per-dialect WER and Mean WER.

Needs in same folder:
  test_logits.pkl, lm_4gram.bin, dialect_lms/lm_<dialect>.bin, best_lm_settings.json

Run: CPU, ~15-20 min.
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

with open(f"{HERE}/best_lm_settings.json") as f:
    best_settings = json.load(f)

with open(f"{HERE}/test_logits.pkl", "rb") as f:
    td = pickle.load(f)

test_df = pd.read_excel("/content/drive/MyDrive/dataset/test.xlsx")
test_df["dialect"] = test_df["file_name"].apply(
    lambda fn: (re.search(r"test_([a-z]+)_", str(fn)).group(1)
                if re.search(r"test_([a-z]+)_", str(fn)) else None)
)

test_by_dialect = {}
for d in test_df["dialect"].dropna().unique():
    idx = test_df.index[test_df["dialect"] == d].tolist()
    test_by_dialect[d] = {
        "logits": [td["logits"][i] for i in idx],
        "refs":   [td["refs"][i]   for i in idx],
    }

warnings.filterwarnings("ignore")
logging.getLogger("pyctcdecode").setLevel(logging.ERROR)

cache = {}
def get_decoder(lm_path, alpha, beta):
    key = (lm_path, alpha, beta)
    if key not in cache:
        cache[key] = build_ctcdecoder(labels=vocab_list, kenlm_model_path=lm_path, alpha=alpha, beta=beta)
    return cache[key]

results = {}
all_greedy_refs, all_greedy_preds = [], []
all_winner_refs, all_winner_preds = [], []

print(f"{'Dialect':<14} {'N':>5} {'Greedy':>9} {'+Global':>10} {'+Dialect':>11} {'+Winner':>10}")
print("-" * 65)
for dialect in sorted(test_by_dialect.keys()):
    tlogits = test_by_dialect[dialect]["logits"]
    trefs   = test_by_dialect[dialect]["refs"]
    s = best_settings[dialect]

    g_decoder = get_decoder(GLOBAL_LM, s["global"]["alpha"], s["global"]["beta"])
    d_decoder = get_decoder(f"{DIALECT_LM_DIR}/lm_{dialect}.bin", s["dialect"]["alpha"], s["dialect"]["beta"])

    greedy, global_lm, dialect_lm = [], [], []
    for lg in tlogits:
        lf32 = lg.astype(np.float32)
        greedy.append(processor.batch_decode(np.argmax(lf32, axis=-1)[None], skip_special_tokens=True)[0])
        global_lm.append(g_decoder.decode(lf32, beam_width=100))
        dialect_lm.append(d_decoder.decode(lf32, beam_width=100))

    greedy     = [p if p.strip() else " " for p in greedy]
    global_lm  = [p if p.strip() else " " for p in global_lm]
    dialect_lm = [p if p.strip() else " " for p in dialect_lm]

    winner_preds = dialect_lm if s["winner"] == "dialect" else global_lm

    g_wer  = round(100 * jiwer.wer(trefs, greedy), 1)
    gl_wer = round(100 * jiwer.wer(trefs, global_lm), 1)
    dl_wer = round(100 * jiwer.wer(trefs, dialect_lm), 1)
    w_wer  = round(100 * jiwer.wer(trefs, winner_preds), 1)

    results[dialect] = {
        "n": len(trefs), "greedy": g_wer, "global": gl_wer,
        "dialect": dl_wer, "winner": w_wer, "winner_choice": s["winner"],
    }
    all_greedy_refs.extend(trefs);  all_greedy_preds.extend(greedy)
    all_winner_refs.extend(trefs);  all_winner_preds.extend(winner_preds)

    print(f"{dialect:<14} {len(trefs):>5} {g_wer:>8.1f}% {gl_wer:>9.1f}% {dl_wer:>10.1f}% {w_wer:>9.1f}%")

mean_greedy = round(100 * jiwer.wer(all_greedy_refs, all_greedy_preds), 1)
mean_winner = round(100 * jiwer.wer(all_winner_refs, all_winner_preds), 1)
print("-" * 65)
print(f"{'MEAN WER':<14} {len(all_greedy_refs):>5} {mean_greedy:>8.1f}% {'':>10} {'':>11} {mean_winner:>9.1f}%")

with open(f"{HERE}/test_results_final.json", "w") as f:
    json.dump({"mean_greedy": mean_greedy, "mean_winner": mean_winner, "per_dialect": results}, f, indent=2)
print(f"\nSaved: {HERE}/test_results_final.json")
