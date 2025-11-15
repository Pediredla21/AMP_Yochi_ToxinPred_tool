import pandas as pd
import numpy as np
import itertools
from pathlib import Path

# ───────────────────────── Paths ─────────────────────────
PROCESSED_PATH = Path("data/processed/")

# Amino acids (standard 20)
AA = "ACDEFGHIKLMNPQRSTVWY"
AA2 = [a + b for a, b in itertools.product(AA, repeat=2)]  # 400 dipeptides

def aac(seq: str) -> np.ndarray:
    """Amino acid composition (20 features)."""
    seq = str(seq).strip().upper()
    L = max(len(seq), 1)
    counts = {a: 0 for a in AA}
    for ch in seq:
        if ch in counts:
            counts[ch] += 1
    return np.array([counts[a] / L for a in AA], dtype=float)

def dpc(seq: str) -> np.ndarray:
    """Dipeptide composition (400 features)."""
    seq = str(seq).strip().upper()
    L = len(seq)
    counts = {d: 0 for d in AA2}
    for i in range(L - 1):
        di = seq[i:i+2]
        if di in counts:
            counts[di] += 1
    denom = max(L - 1, 1)
    return np.array([counts[d] / denom for d in AA2], dtype=float)

def featurize(seq: str) -> np.ndarray:
    """Combine AAC + DPC = 420 features."""
    return np.concatenate([aac(seq), dpc(seq)])

if __name__ == "__main__":
    # Load processed train/test tables from step 00
    train = pd.read_csv(PROCESSED_PATH / "train.csv")
    test  = pd.read_csv(PROCESSED_PATH / "test.csv")

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    # Extract features
    X_train = np.vstack([featurize(s) for s in train["seq"]])
    y_train = train["label"].values

    X_test  = np.vstack([featurize(s) for s in test["seq"]])
    y_test  = test["label"].values

    # Save as numpy arrays
    np.save(PROCESSED_PATH / "X_train.npy", X_train)
    np.save(PROCESSED_PATH / "y_train.npy", y_train)
    np.save(PROCESSED_PATH / "X_test.npy",  X_test)
    np.save(PROCESSED_PATH / "y_test.npy",  y_test)

    # Optional: CSV versions (for inspection / model zoo)
    pd.DataFrame(X_train).to_csv(PROCESSED_PATH / "X_train.csv", index=False)
    pd.DataFrame(X_test ).to_csv(PROCESSED_PATH / "X_test.csv",  index=False)

    print("\n✅ Features extracted and saved in data/processed/")
    print("Train features:", X_train.shape, " | Test features:", X_test.shape)
