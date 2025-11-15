import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

PROCESSED_PATH = Path("data/processed/")
MODEL_PATH = Path("models/")

if __name__ == "__main__":
    # Load tuned model if exists, otherwise baseline AAC+DPC model
    try:
        clf = joblib.load(MODEL_PATH / "extratrees_tuned.joblib")
        print("Loaded tuned model (extratrees_tuned.joblib).")
    except Exception:
        clf = joblib.load(MODEL_PATH / "extratrees_aac_dpc.joblib")
        print("Loaded baseline model (extratrees_aac_dpc.joblib).")

    # Feature names: 20 AAC + 400 DPC
    AA = "ACDEFGHIKLMNPQRSTVWY"
    AA2 = [a + b for a in AA for b in AA]
    features = [f"AAC_{a}" for a in AA] + [f"DPC_{d}" for d in AA2]

    importances = clf.feature_importances_
    df_imp = pd.DataFrame({"feature": features, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False)

    # Save CSV
    out_csv = PROCESSED_PATH / "feature_importance.csv"
    df_imp.to_csv(out_csv, index=False)
    print(f"✅ Saved feature importances to {out_csv}")

    # Plot top 20
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp.head(20), x="importance", y="feature")
    plt.title("Top 20 Features Driving Toxicity (ExtraTrees AAC+DPC)")
    plt.tight_layout()
    out_png = PROCESSED_PATH / "top20_feature_importance.png"
    plt.savefig(out_png)
    plt.close()

    print(f"📊 Saved plot: {out_png}")
