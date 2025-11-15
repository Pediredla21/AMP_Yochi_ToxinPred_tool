import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, accuracy_score,
    confusion_matrix, classification_report
)
import joblib

P = Path("data/processed/")
M = Path("models/")

if __name__ == "__main__":
    # ───────────────────────── Load data ─────────────────────────
    X_train = np.load(P / "X_train.npy")
    y_train = np.load(P / "y_train.npy")
    X_test  = np.load(P / "X_test.npy")
    y_test  = np.load(P / "y_test.npy")

    print("Train:", X_train.shape, " Test:", X_test.shape)

    # Split off a validation set from training (stratified)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    # ───────────────────────── Load model ─────────────────────────
    try:
        clf = joblib.load(M / "extratrees_tuned.joblib")
        print("Loaded tuned model (extratrees_tuned.joblib).")
    except Exception:
        clf = joblib.load(M / "extratrees_aac_dpc.joblib")
        print("Loaded baseline model (extratrees_aac_dpc.joblib).")

    # Fit on training split (not validation)
    clf.fit(X_tr, y_tr)

    # ───────────────────────── Find best threshold on validation ─────────────────────────
    val_prob = clf.predict_proba(X_val)[:, 1]

    best_t, best_mcc = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):  # 0.10, 0.11, ..., 0.90
        pred = (val_prob >= t).astype(int)
        mcc = matthews_corrcoef(y_val, pred)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t

    print(f"\nBest threshold on validation (MCC): t={best_t:.3f}, MCC={best_mcc:.4f}")

    # ───────────────────────── Evaluate on independent test set ─────────────────────────
    test_prob = clf.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_t).astype(int)

    print("\n📊 Test set with optimized threshold")
    print("Accuracy:", accuracy_score(y_test, test_pred))
    print("AUROC:", roc_auc_score(y_test, test_prob))  # AUROC independent of threshold
    print("MCC:", matthews_corrcoef(y_test, test_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, test_pred))
    print(classification_report(y_test, test_pred, target_names=["Non-toxic", "Toxic"]))
