import numpy as np
import pandas as pd
import itertools
import streamlit as st
import joblib
from pathlib import Path
from html import escape

# ───────────────────────── Path helpers ─────────────────────────
def here() -> Path:
    return Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    here().parent / "models" / "extratrees_tuned_calibrated.joblib",
    here().parent / "models" / "extratrees_tuned.joblib",
    here().parent / "models" / "extratrees_aac_dpc.joblib",
]

# ───────────────────────── Load model ─────────────────────────
clf, loaded_path = None, None
for p in MODEL_CANDIDATES:
    if p.exists():
        clf = joblib.load(p)
        loaded_path = p
        break

if clf is None:
    st.error("No model found in models/. Please train first (run 03, 05, 08).")
    st.stop()

IS_CALIBRATED = "calibrated" in str(loaded_path).lower()

# ───────────────────────── Feature functions ─────────────────────────
AA = "ACDEFGHIKLMNPQRSTVWY"
AA2 = [a + b for a, b in itertools.product(AA, repeat=2)]

def aac(seq: str) -> np.ndarray:
    seq = seq.strip().upper()
    L = max(len(seq), 1)
    counts = {a: 0 for a in AA}
    for ch in seq:
        if ch in counts:
            counts[ch] += 1
    return np.array([counts[a] / L for a in AA], dtype=float)

def dpc(seq: str) -> np.ndarray:
    seq = seq.strip().upper()
    L = len(seq)
    counts = {d: 0 for d in AA2}
    for i in range(L - 1):
        di = seq[i:i+2]
        if di in counts:
            counts[di] += 1
    denom = max(L - 1, 1)
    return np.array([counts[d] / denom for d in AA2], dtype=float)

def featurize(seq: str) -> np.ndarray:
    return np.concatenate([aac(seq), dpc(seq)])[None, :]

# ───────────────────────── Motif table for pattern hints ─────────────────────────
MOTIF_PATH = here().parent / "data" / "processed" / "motif_logodds.csv"
motif_lut = {}
if MOTIF_PATH.exists():
    dfm = pd.read_csv(MOTIF_PATH)
    motif_lut = {(int(r.k), r.motif): float(r.logodds) for _, r in dfm.iterrows()}

def kmers(s: str, k: int):
    s = s.strip().upper()
    return [s[i:i+k] for i in range(len(s)-k+1)] if len(s) >= k else []

def top_motif_hits(seq: str, ks=(2,3), topn=8):
    if not motif_lut:
        return []
    bag = {}
    seq = seq.strip().upper()
    for k in ks:
        for m in kmers(seq, k):
            lo = motif_lut.get((k, m), 0.0)
            if lo > 0:
                bag[(k, m)] = bag.get((k, m), 0.0) + lo
    rows = [{"k": k, "motif": m, "score": round(lo, 3)} for (k, m), lo in bag.items()]
    return sorted(rows, key=lambda r: r["score"], reverse=True)[:topn]

def valid_seq(s: str) -> bool:
    return (3 <= len(s) <= 35) and all(ch in AA for ch in s)

# ───────────────────────── Streamlit page config & styling ─────────────────────────
st.set_page_config(
    page_title="AMP Yochi — Peptide Toxicity Checker",
    page_icon="🧪",
    layout="wide"
)

# Custom CSS for “Lovable-style” UI
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background: radial-gradient(circle at top left, #1f2933 0, #050814 45%, #020308 100%);
        color: #f9fafb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    }

    /* Main app container */
    .main-block {
        max-width: 1100px;
        margin: 0 auto;
        padding-top: 0.5rem;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(12, 17, 28, 0.85);
        border-radius: 18px;
        padding: 18px 20px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.55);
    }

    .pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .pill-safe {
        background: rgba(52, 211, 153, 0.16);
        color: #6ee7b7;
        border: 1px solid rgba(52, 211, 153, 0.45);
    }
    .pill-toxic {
        background: rgba(248, 113, 113, 0.16);
        color: #fecaca;
        border: 1px solid rgba(248, 113, 113, 0.45);
    }

    .metric-big {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
    }
    .metric-label {
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9ca3af;
        margin-bottom: 0.35rem;
    }
    .section-title {
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9ca3af;
        margin-bottom: 0.4rem;
    }
    .app-title {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-top: 0.25rem;
    }
    .model-chip {
        display: inline-flex;
        align-items: center;
        padding: 3px 9px;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.5);
        font-size: 0.72rem;
        color: #e5e7eb;
        gap: 6px;
    }
    .model-chip span.icon {
        font-size: 0.9rem;
    }
    .divider {
        border-bottom: 1px dashed rgba(148, 163, 184, 0.35);
        margin: 0.6rem 0 0.9rem 0;
    }

    .seq-box {
        font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 0.85rem;
        padding: 6px 10px;
        border-radius: 10px;
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(75, 85, 99, 0.9);
        color: #e5e7eb;
    }

    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────── Layout ─────────────────────────
st.markdown("<div class='main-block'>", unsafe_allow_html=True)

# Header
st.markdown(
    f"""
    <div class="glass-card" style="margin-bottom: 1rem;">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;">
        <div>
          <div class="app-title">🧪 AMP Yochi — Peptide Toxicity Checker</div>
          <div class="subtitle">
            Short antimicrobial peptide safety checker inspired by ToxinPred&nbsp;3.0, 
            trained on AAC+DPC features from curated toxic vs non-toxic peptides.
          </div>
        </div>
        <div style="text-align:right;">
          <div class="model-chip">
            <span class="icon">⚙️</span>
            <span>{'Calibrated ExtraTrees' if IS_CALIBRATED else 'ExtraTrees (standard)'}</span>
          </div>
          <div style="font-size:0.7rem; color:#9ca3af; margin-top:4px;">
            Model file: <code style="font-size:0.7rem;">{escape(loaded_path.name)}</code>
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 1])

# ───────────────────────── Left: Input + controls ─────────────────────────
with left:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Input peptide</div>", unsafe_allow_html=True)

    default_seq = "GCPWMPWC"
    seq = st.text_input(
        label="",
        value=default_seq,
        max_chars=50,
        help=f"Sequence length 3–35; allowed letters: {AA}",
    ).strip().upper()

    st.markdown(
        f"<div class='seq-box'>{escape(seq) if seq else '—'}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        show_patterns = st.checkbox("Show motif pattern hints", value=True)
    with col_b:
        mutate_to = st.selectbox("Swap residue to", options=list("AGSTN"), index=0)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔍 Run toxicity prediction", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────────── Right: Prediction + visuals ─────────────────────────
with right:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    if not run_btn:
        st.markdown(
            "<div style='color:#9ca3af; font-size:0.9rem;'>"
            "Enter a peptide and click <b>Run toxicity prediction</b> to see the model output."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        if not valid_seq(seq):
            st.error(f"Please enter a valid sequence (3–35 letters, only from: {AA}).")
        else:
            X = featurize(seq)
            prob = float(clf.predict_proba(X)[0, 1])
            label = "Toxic" if prob >= 0.5 else "Non-toxic"

            # Top row: label + probability
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("<div class='metric-label'>Prediction</div>", unsafe_allow_html=True)
                pill_class = "pill-toxic" if label == "Toxic" else "pill-safe"
                st.markdown(
                    f"<div class='pill {pill_class}'>{label}</div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown("<div class='metric-label'>Toxicity probability</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-big'>{prob:.3f}</div>", unsafe_allow_html=True)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # Residue-level sensitivity
            st.markdown("<div class='section-title'>Residue ribbon — single-letter swaps</div>", unsafe_allow_html=True)

            base_prob = prob
            contrib = np.zeros(len(seq))
            for i in range(len(seq)):
                mutated = seq[:i] + mutate_to + seq[i+1:]
                p2 = float(clf.predict_proba(featurize(mutated))[0, 1])
                contrib[i] = base_prob - p2  # positive => current residue raises risk

            max_abs = np.max(np.abs(contrib)) or 1.0
            norm = contrib / max_abs

            def color_for(v):
                a = min(1.0, max(0.0, abs(float(v))))
                if v >= 0:
                    r, g, b = (248, 113, 113)   # red-ish
                else:
                    r, g, b = (52, 211, 153)    # green-ish
                w = 0.35 + 0.65 * (1 - a)
                r = int(r * (1 - w) + 255 * w)
                g = int(g * (1 - w) + 255 * w)
                b = int(b * (1 - w) + 255 * w)
                return f"rgb({r},{g},{b})"

            cells = []
            for i, ch in enumerate(seq):
                bg = color_for(norm[i])
                title = f"pos {i+1} | ΔProb={contrib[i]:+.3f} if {ch}->{mutate_to}"
                cell = (
                    f"<div style='display:inline-block; text-align:center; margin:3px;'>"
                    f"<div title='{escape(title)}' style='width:30px; height:30px; line-height:30px;"
                    f"border-radius:10px; background:{bg}; color:#000; border:1px solid rgba(15,23,42,0.7);"
                    f"font-family:monospace; font-size:0.85rem;'>"
                    f"{escape(ch)}</div>"
                    f"<div style='color:#9ca3af; font-size:11px; margin-top:2px; font-family:monospace;'>"
                    f"{contrib[i]:+.2f}</div></div>"
                )
                cells.append(cell)

            st.markdown("".join(cells), unsafe_allow_html=True)
            st.caption(
                f"ΔProb = original_prob − prob_after_swapping that residue to **{mutate_to}**. "
                "Positive values mean the current residue appears riskier."
            )

            # Motif-based hints
            if show_patterns:
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Motif-based pattern hints</div>", unsafe_allow_html=True)
                hits = top_motif_hits(seq, ks=(2,3), topn=8)
                if hits:
                    st.dataframe(pd.DataFrame(hits), use_container_width=True, height=240)
                    st.caption(
                        "Motifs with higher score are more enriched in toxic peptides "
                        "in your training set (based on 2-mer & 3-mer log-odds)."
                    )
                else:
                    st.info("No enriched dipeptide/tripeptide motifs detected for this sequence.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
