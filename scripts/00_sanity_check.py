import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ───────────────────────── Paths ─────────────────────────
RAW_PATH = Path("data/raw/")
PROCESSED_PATH = Path("data/processed/")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# ───────────────────────── Load raw files ─────────────────────────
train_pos = pd.read_csv(RAW_PATH / "train_pos.csv", header=None, names=["seq"])
train_neg = pd.read_csv(RAW_PATH / "train_neg.csv", header=None, names=["seq"])
test_pos  = pd.read_csv(RAW_PATH / "test_pos.csv",  header=None, names=["seq"])
test_neg  = pd.read_csv(RAW_PATH / "test_neg.csv",  header=None, names=["seq"])

# ───────────────────────── Add labels ─────────────────────────
train_pos["label"] = 1
train_neg["label"] = 0
test_pos["label"]  = 1
test_neg["label"]  = 0

# ───────────────────────── Merge train & test ─────────────────────────
train = pd.concat([train_pos, train_neg], ignore_index=True)
test  = pd.concat([test_pos, test_neg], ignore_index=True)

# Save processed datasets
train.to_csv(PROCESSED_PATH / "train.csv", index=False)
test.to_csv(PROCESSED_PATH / "test.csv", index=False)

print("✅ Saved train.csv and test.csv in data/processed/")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ───────────────────────── Basic Stats ─────────────────────────
train["length"] = train["seq"].str.len()
test["length"] = test["seq"].str.len()

print("\nTrain length summary:\n", train["length"].describe())
print("\nTest length summary:\n", test["length"].describe())

# duplicates
print("\nDuplicate sequences (train):", train["seq"].duplicated().sum())
print("Duplicate sequences (test):",  test["seq"].duplicated().sum())

# ───────────────────────── Plotting ─────────────────────────
# 1 — Sequence length distribution
plt.figure(figsize=(8,4))
train["length"].hist(bins=30)
plt.xlabel("Peptide length")
plt.ylabel("Count")
plt.title("Train Sequence Length Distribution")
plt.tight_layout()
plt.savefig(PROCESSED_PATH / "train_length_distribution.png")
plt.close()

# 2 — Amino acid frequency
aa_counts = {}
for seq in train["seq"]:
    for aa in seq:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1

aa_df = (
    pd.DataFrame.from_dict(aa_counts, orient="index", columns=["count"])
      .sort_values("count", ascending=False)
)

plt.figure(figsize=(10,4))
aa_df.plot(kind="bar", legend=False)
plt.xlabel("Amino Acid")
plt.ylabel("Count")
plt.title("Amino Acid Frequency (Train)")
plt.tight_layout()
plt.savefig(PROCESSED_PATH / "train_amino_acid_freq.png")
plt.close()

print("\n📊 Saved plots:")
print("- data/processed/train_length_distribution.png")
print("- data/processed/train_amino_acid_freq.png")
print("\n🎉 Sanity check complete!")

