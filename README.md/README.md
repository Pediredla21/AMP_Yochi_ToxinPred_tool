<!-- ========================================================= -->
<!--                   AMP YOCHI – README                      -->
<!-- ========================================================= -->

<p align="center">
  <img src="https://svg-banners.vercel.app/api?type=origin&text1=AMP%20YOCHI%20⚡%20Peptide%20Toxicity%20Predictor&width=1200&height=300" />
</p>

<h2 align="center">🔬 Residue-Level Explainability • Swap Suggestions • Calibrated Toxicity Scores</h2>

<p align="center">
  Built by <b>Bhavani Pediredla</b>  
</p>

<br>

---

# AMP Yochi – Peptide Toxicity Prediction & Residue-Level Explanations

AMP Yochi is an **end-to-end machine learning system** that predicts peptide toxicity and explains **WHY** a sequence looks risky — at the **residue level**.

 **Input:** a short amino-acid sequence  
 **Output:** toxicity probability + residue-level explanations  
 **Goal:** help scientists design safer peptides, not just classify sequences

---

# Why This Project Exists

Most existing toxicity tools output only:

 *“toxic / non-toxic”*

Scientists need **much more**:

- *“Which residues are risky?”*  
- *“If I change a residue, will it reduce toxicity?”*

 **AMP Yochi solves this by providing residue-wise contributions and swap suggestions.**

---

#  What Makes AMP Yochi Unique

✔ **Residue Ribbon** – shows how each residue contributes to toxicity  
✔ **Residue Swap Suggestions** – shows safer alternative residues  
✔ **Calibrated Probabilities** – confidence-aware toxicity prediction  
✔ **Hybrid Toxicity Logic** – AAC + DPC + motif-aware scoring  
✔ **Full ML product** – preprocessing → modeling → evaluation → API → UI  

Not just a model — a complete scientific ML tool.

---

# Features (High-Level)

## 🔬 Core ML
- **ExtraTreesClassifier** trained on:
  - AAC (20 amino-acid composition features)
  - DPC (400 dipeptide composition features)
- Hyperparameter tuning using cross-validation
- **Probability calibration** using isotonic regression
- Motif-aware hybrid scoring for Cys-rich toxic motifs

## Explainability
- Feature importance for AAC/DPC
- **Residue Ribbon** → per-residue contribution
- **Swap Suggestion Engine** → suggests residue replacements that reduce toxicity

##  Interfaces
### **FastAPI Backend**
