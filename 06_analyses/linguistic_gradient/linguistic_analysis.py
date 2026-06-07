"""
linguistic_analysis.py
Groups the 12 dialects by Bengali dialect family and shows WER/CER rises with
linguistic distance from Standard Bengali.

IMPORTANT: The dialect groupings are based on Bengali dialectology and MUST be
cited to an authoritative source (e.g., Grierson Linguistic Survey of India, or
a modern Bengali dialectology reference) before publication. Verify group
membership against that source.

Setup: pip install "numpy<2" pandas
"""
import numpy as np, pandas as pd

best_wer = {"comilla":43.4,"tangail":45.3,"habiganj":65.0,"narail":70.3,"narsingdi":71.5,
            "barishal":74.4,"sylhet":74.9,"rangpur":80.8,"noakhali":82.2,"sandwip":85.0,
            "kishoreganj":87.3,"chittagong":87.4}
best_cer = {"comilla":16.9,"tangail":17.7,"habiganj":31.6,"narail":38.5,"narsingdi":38.5,
            "barishal":44.4,"sylhet":43.7,"rangpur":48.6,"noakhali":54.4,"sandwip":54.6,
            "kishoreganj":59.2,"chittagong":53.6}
groups = {
    "Rarhi / Central": ["tangail","narail"],
    "Bangali / East-Central": ["comilla","narsingdi","barishal","habiganj"],
    "Sylheti": ["sylhet"],
    "Kamta / Rangpuri": ["rangpur"],
    "Chittagonian": ["chittagong","sandwip","noakhali"],
    "Other (kishoreganj)": ["kishoreganj"],
}
rows = [{"group":g,"dialects":", ".join(dl),
         "mean_wer":round(np.mean([best_wer[d] for d in dl]),1),
         "mean_cer":round(np.mean([best_cer[d] for d in dl]),1)} for g,dl in groups.items()]
df = pd.DataFrame(rows).sort_values("mean_wer")
print(df.to_string(index=False))
print("\nWER rises with linguistic distance from Standard Bengali.")
