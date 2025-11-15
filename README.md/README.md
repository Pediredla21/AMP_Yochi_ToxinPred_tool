
---

#  **About AMP Yochi**
AMP Yochi is an **end-to-end explainable machine learning system** built to predict toxin-like activity in antimicrobial peptides (AMPs) and **explain exactly WHY** a sequence looks toxic.

Instead of just labeling sequences as *toxic / non-toxic*, AMP Yochi provides:

✔ **Residue-level risk visualization**  
✔ **Suggested amino-acid swaps** to reduce toxicity  
✔ **Calibrated probabilities** for confidence  
✔ A fully reproducible ML pipeline  
✔ A FastAPI backend and UI integration  

This makes AMP Yochi not just a model — **a complete peptide-design tool**.

---

#  **Project Demo (Video)**  

---

#  **Why This Project Exists**

Most peptide toxicity tools only answer one question:

>  *“Is this peptide toxic?”*

But in real biological design, scientists need much deeper answers:

- **Which residues are contributing to toxicity?**  
- **If I mutate one residue, will toxicity drop?**  
- **How confident is the prediction?**  

**AMP Yochi fills this critical gap** by providing interpretability, actionable suggestions, and calibrated toxicity probability.

---

#  **Key Features**

## 1. Explainable Machine Learning
- ExtraTreesClassifier trained on:
  - **Amino Acid Composition (AAC)** – 20 features  
  - **Dipeptide Composition (DPC)** – 400 features  
- Motif-aware scoring for:
  - Cys-rich motifs  
  - Known toxic dipeptides

##  2. Residue Ribbon (Explainability)
A heatmap-like per-residue bar that shows:

- Which positions contribute most to toxicity  
- Local risk patterns  
- The “shape” of toxicity within the sequence  

## 🔁 3. Residue Swap Suggestions
Based on model gradients + residue frequencies:

- Suggests **1-letter substitutions** to reduce toxicity
- Predicts new toxicity probability after swap
- Only gives **biologically reasonable** substitutions

## 📈 4. Probability Calibration
- Uses **isotonic regression** for calibrated predicted probabilities  
- Ensures the toxicity score is **trustworthy**, not arbitrary  

## 5. API + UI
- FastAPI backend → `/predict`
- Streamlit demo UI
- Modern React UI (Lovable) with ribbon + swap visualization

---


---

#  **Model Performance & Insights**

##  Amino Acid Frequency (Training Data)
- Most common: **C, L, G**  
- Realistic AMP composition  
- Gives confidence in training data quality

##  Top Features Driving Toxicity (AAC + DPC)
- **AAC_C** (Cysteine content) – highest signal  
- **DPC_CC** (Cys–Cys) – strong toxic motif  
- Charged residues: K, D, E, R  
- DPC patterns: CS, KK, PC, LP, CG  

### Interpretation:
Cysteine-rich peptides tend to form **stable disulfide-bonded structures** → more toxic.

---

## Confusion Matrix (Held-Out Test Set)

|                  | Pred Non-toxic | Pred Toxic |
|------------------|----------------|------------|
| **True Non-toxic** | 1013 | 91 |
| **True Toxic**     | 143 | 961 |

### Metrics:
- Accuracy: **89.4%**  
- Precision (toxic): **91%**  
- Recall (toxic): **87%**  
- AUROC: **0.95**  

### Summary:
The model is slightly conservative (prefers fewer false positives), which is ideal for real-world design.

---

#  **How to Run the Project**

##  Clone Repo & Setup Environment
```bash
git clone https://github.com/<your-username>/AMP_Yochi.git
cd AMP_Yochi

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt





