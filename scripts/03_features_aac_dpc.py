import numpy as np
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ───────────────────────── Paths ─────────────────────────
PROCESSED_PATH = Path("data/processed/")
MODEL_PATH = Path("models/")
MODEL_PATH.mkdir(exist_ok=True)

if __name__ == "__main__":
    # Load features
    X_train = np.load(PROCESSED_PATH / "X_train.npy")
    y_train = np.load(PROCESSED_PATH / "y_train.npy")
    X_test  = np.load(PROCESSED_PATH / "X_test.npy")
    y_test  = np.load(PROCESSED_PATH / "y_test.npy")

    print("Train:", X_train.shape, " Test:", X_test.shape)

    # Train ExtraTrees
    clf = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    # Save model
    joblib.dump(clf, MODEL_PATH / "extratrees_aac_dpc.joblib")
    print("✅ Model saved in models/extratrees_aac_dpc.joblib")

    # Predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)

    print("\n📊 Test Performance (ExtraTrees, AAC+DPC)")
    print("Accuracy:", acc)
    print("AUROC:", auc)
    print("MCC:", mcc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-toxic","Toxic"],
        yticklabels=["Non-toxic","Toxic"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (ExtraTrees AAC+DPC)")
    plt.tight_layout()
    plt.savefig(PROCESSED_PATH / "extratrees_confusion_matrix.png")
    plt.close()

    print("📊 Confusion matrix saved in data/processed/extratrees_confusion_matrix.png")

    # ROC Curve
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_estimator(clf, X_test, y_test)
    plt.title("ROC Curve (ExtraTrees AAC+DPC)")
    plt.tight_layout()
    plt.savefig(PROCESSED_PATH / "extratrees_roc.png")
    plt.close()

    print("📊 ROC curve saved in data/processed/extratrees_roc.png")
