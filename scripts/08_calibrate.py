import numpy as np
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
from joblib import load, dump

P = Path("data/processed/")
M = Path("models/")

if __name__ == "__main__":
    # ───────────────────────── Load data ─────────────────────────
    X_train = np.load(P / "X_train.npy")
    y_train = np.load(P / "y_train.npy")
    X_test  = np.load(P / "X_test.npy")
    y_test  = np.load(P / "y_test.npy")

    print("Train:", X_train.shape, " Test:", X_test.shape)

    # ───────────────────────── Load base model ─────────────────────────
    try:
        base = load(M / "extratrees_tuned.joblib")
        print("Loaded tuned model (extratrees_tuned.joblib).")
    except Exception:
        base = load(M / "extratrees_aac_dpc.joblib")
        print("Loaded baseline model (extratrees_aac_dpc.joblib).")

    # ───────────────────────── Set up calibrator ─────────────────────────
    # IMPORTANT: pass the estimator as *positional* arg (no keyword),
    # so it works with both older and newer sklearn versions.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cal = CalibratedClassifierCV(base, cv=cv, method="isotonic")

    print("Fitting calibrated model (this may take a bit)...")
    cal.fit(X_train, y_train)

    # ───────────────────────── Save calibrated model ─────────────────────────
    out_model = M / "extratrees_tuned_calibrated.joblib"
    dump(cal, out_model)
    print(f"✅ Saved calibrated model: {out_model}")

    # ───────────────────────── Evaluate on test set ─────────────────────────
    p_test = cal.predict_proba(X_test)[:, 1]
    # You can later change 0.5 → 0.4 (threshold from 07) if you want
    y_pred = (p_test >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, p_test)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)

    print("\n📊 Calibrated ExtraTrees Performance on Test Set")
    print("Accuracy:", acc)
    print("AUROC:", auc)
    print("MCC:", mcc)
    print("Confusion matrix:\n", cm)
    print(classification_report(y_test, y_pred, target_names=["Non-toxic", "Toxic"]))
