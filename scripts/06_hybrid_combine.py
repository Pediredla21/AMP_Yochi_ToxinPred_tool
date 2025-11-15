import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from joblib import load

# ───────────────────────── Paths ─────────────────────────
PROCESSED_PATH = Path("data/processed/")
MODEL_PATH = Path("models/")

def load_arrays():
    """
    Load:
    - X_train, y_train, X_test, y_test from .npy
    - motif_score for train/test from *_motif.csv
    - Z-normalize motif scores using train stats
    """
    # AAC+DPC feature arrays
    Xtr = np.load(PROCESSED_PATH / "X_train.npy")
    ytr = np.load(PROCESSED_PATH / "y_train.npy")
    Xte = np.load(PROCESSED_PATH / "X_test.npy")
    yte = np.load(PROCESSED_PATH / "y_test.npy")

    # motif scores computed in 04b_train_compare_ext.py
    train_motif = pd.read_csv(PROCESSED_PATH / "train_motif.csv")["motif_score"].values
    test_motif  = pd.read_csv(PROCESSED_PATH / "test_motif.csv")["motif_score"].values

    # Z-normalize based on train distribution
    m_mean = train_motif.mean()
    m_std  = train_motif.std() + 1e-9
    mtr_z = (train_motif - m_mean) / m_std
    mte_z = (test_motif  - m_mean) / m_std

    return Xtr, ytr, Xte, yte, mtr_z, mte_z

if __name__ == "__main__":
    # ───────────────────────── Load model ─────────────────────────
    try:
        model = load(MODEL_PATH / "extratrees_tuned.joblib")
        print("Loaded tuned model (extratrees_tuned.joblib).")
    except Exception:
        model = load(MODEL_PATH / "extratrees_aac_dpc.joblib")
        print("Loaded baseline model (extratrees_aac_dpc.joblib).")

    Xtr, ytr, Xte, yte, mtr, mte = load_arrays()

    # model probabilities
    p_tr = model.predict_proba(Xtr)[:, 1]
    p_te = model.predict_proba(Xte)[:, 1]

    # sigmoid for motif scores
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))

    best_alpha, best_mcc, best_auc = None, -1.0, None

    # ───────────────────────── Search best α on train ─────────────────────────
    for a in np.linspace(0.0, 1.0, 21):  # 0.00, 0.05, ..., 1.00
        comb_tr = a * p_tr + (1 - a) * sig(mtr)
        pred_tr = (comb_tr >= 0.5).astype(int)
        mcc = matthews_corrcoef(ytr, pred_tr)
        auc = roc_auc_score(ytr, comb_tr)
        if mcc > best_mcc:
            best_alpha, best_mcc, best_auc = a, mcc, auc

    print(f"Best alpha on train: a={best_alpha:.2f} | MCC={best_mcc:.4f} | AUROC={best_auc:.4f}")

    # ───────────────────────── Evaluate on test ─────────────────────────
    comb_te = best_alpha * p_te + (1 - best_alpha) * sig(mte)
    pred_te = (comb_te >= 0.5).astype(int)

    test_auc = roc_auc_score(yte, comb_te)
    test_mcc = matthews_corrcoef(yte, pred_te)
    print(f"HYBRID TEST  AUROC={test_auc:.4f} | MCC={test_mcc:.4f}")

    # ───────────────────────── Save predictions ─────────────────────────
    out_df = pd.DataFrame({
        "prob_hybrid": comb_te,
        "pred_hybrid": pred_te,
        "label": yte
    })
    out_path = PROCESSED_PATH / "test_hybrid_preds.csv"
    out_df.to_csv(out_path, index=False)
    print(f"✅ Saved hybrid test predictions to: {out_path}")
