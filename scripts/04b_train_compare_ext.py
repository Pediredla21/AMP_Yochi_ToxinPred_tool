import pandas as pd
import math
from collections import Counter
from pathlib import Path

PROCESSED_PATH = Path("data/processed/")

def kmers(s: str, k: int):
    s = str(s).strip().upper()
    return [s[i:i+k] for i in range(len(s)-k+1)] if len(s) >= k else []

def compute_enrichment(train_csv: str = "data/processed/train.csv", ks=(2,3)):
    df = pd.read_csv(train_csv)
    tox = df[df.label == 1]["seq"].tolist()
    nto = df[df.label == 0]["seq"].tolist()

    rows = []
    for k in ks:
        ct_t = Counter(m for s in tox for m in kmers(s, k))
        ct_n = Counter(m for s in nto for m in kmers(s, k))

        allm = set(ct_t) | set(ct_n)
        # Laplace smoothing
        tot_t = sum(ct_t.values()) + len(allm)
        tot_n = sum(ct_n.values()) + len(allm)

        for m in allm:
            pt = (ct_t[m] + 1) / tot_t
            pn = (ct_n[m] + 1) / tot_n
            rows.append({"k": k, "motif": m, "logodds": math.log(pt / pn)})

    out_path = PROCESSED_PATH / "motif_logodds.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"✅ Saved motif log-odds table: {out_path}")

def score_sequences(in_csv: str, out_csv: str, ks=(2,3)):
    df = pd.read_csv(in_csv)
    table = pd.read_csv(PROCESSED_PATH / "motif_logodds.csv")
    # lookup: (k, motif) -> logodds
    lut = {(int(r.k), r.motif): float(r.logodds) for _, r in table.iterrows()}

    def score(seq: str) -> float:
        seq = str(seq).strip().upper()
        s = 0.0
        for k in ks:
            for m in kmers(seq, k):
                s += lut.get((k, m), 0.0)
        return s

    df["motif_score"] = df["seq"].apply(score)
    df.to_csv(out_csv, index=False)
    print(f"✅ Scored motif_score to: {out_csv}")

if __name__ == "__main__":
    compute_enrichment()
    score_sequences("data/processed/train.csv", "data/processed/train_motif.csv")
    score_sequences("data/processed/test.csv",  "data/processed/test_motif.csv")
