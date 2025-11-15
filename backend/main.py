from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path
import itertools

app = FastAPI(title="AMP Yochi API")

# CORS so your frontend can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────── Load model ───────────
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "extratrees_tuned_calibrated.joblib"
MODEL = joblib.load(MODEL_PATH)

AA = "ACDEFGHIKLMNPQRSTVWY"
AA2 = [a + b for a, b in itertools.product(AA, repeat=2)]

def aac(seq: str):
    seq = seq.upper()
    L = max(1, len(seq))
    return np.array([seq.count(a)/L for a in AA], dtype=float)

def dpc(seq: str):
    seq = seq.upper()
    L = max(1, len(seq) - 1)
    return np.array([
        sum(seq[i:i+2] == d for i in range(len(seq)-1))/L
        for d in AA2
    ], dtype=float)

def featurize(seq: str):
    return np.concatenate([aac(seq), dpc(seq)])[None, :]

class PredictRequest(BaseModel):
    sequence: str

@app.post("/predict")
def predict(req: PredictRequest):
    seq = req.sequence.strip().upper()

    if len(seq) < 3 or len(seq) > 35:
        raise HTTPException(status_code=400, detail="Sequence must be 3–35 amino acids")

    if any(c not in AA for c in seq):
        raise HTTPException(status_code=400, detail=f"Invalid characters. Allowed: {AA}")

    X = featurize(seq)
    prob = float(MODEL.predict_proba(X)[0, 1])
    toxic = prob >= 0.5

    return {
        "toxic": toxic,
        "toxicityChance": round(prob, 3),
    }
