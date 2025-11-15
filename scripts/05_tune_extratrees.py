import numpy as np
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
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

    # ───────────────────────── Hyperparameter search space ─────────────────────────
    param_dist = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_features": ["sqrt", "log2", 0.2, 0.4, 0.6, None],
        "min_samples_leaf": [1, 2, 3, 5, 10],
        "max_depth": [None, 20, 40, 60],
        "bootstrap": [True, False]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    clf = ExtraTreesClassifier(random_state=42, n_jobs=-1)

    # ───────────────────────── Run RandomizedSearchCV ─────────────────────────
    search = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=20,          # try 20 random combinations
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    search.fit(X_train, y_train)

    print("\n✅ Best hyperparameters:", search.best_params_)
    print("Best CV AUROC:", search.best_score_)

    # ───────────────────────── Retrain best model ─────────────────────────
    best_model = search.best_estimator_
    best_model.fit(X_train, y_train)

    # Save tuned model
    joblib.dump(best_model, MODEL_PATH / "extratrees_tuned.joblib")
    print("📦 Saved tuned model: models/extratrees_tuned.joblib")

    # ───────────────────────── Evaluate on test set ─────────────────────────
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    print("\n📊 Tuned ExtraTrees Performance on Test Set")
    print("Accuracy:", acc)
    print("AUROC:", auc)
    print("MCC:", mcc)
