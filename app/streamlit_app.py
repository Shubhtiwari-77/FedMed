"""
app/streamlit_app.py
====================
Browser-based Clinical Decision Support System — the demo to show your
examiners. Sliders for the clinical findings, a live ranked differential
diagnosis, and a side-by-side "what if this patient were somewhere else"
panel that makes the context-awareness contribution visible in one screen.

    pip install streamlit
    streamlit run app/streamlit_app.py

Requires a trained model, so run `python src/run_simulation.py` first.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src.cdss import blank_patient, load_model, predict

st.set_page_config(page_title="FedMed-FL CDSS", page_icon="+", layout="wide")


@st.cache_resource
def _model():
    return load_model()


st.title("FedMed-FL — Clinical Decision Support")
st.caption("Global model trained across 3 hospitals by federated learning. "
           "No raw patient record ever left a hospital.")

try:
    model = _model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Patient presentation")

    st.subheader("Context")
    region = st.selectbox("Region / climate", cfg.REGIONS, index=0)
    month = st.slider("Month of presentation", 1, 12, 8)
    st.caption(f"Season: **{cfg.MONTH_TO_SEASON[month]}**")

    st.subheader("Demographics")
    age = st.slider("Age (years)", 1, 95, 31)
    gender = 1 if st.radio("Sex", ["Male", "Female"], horizontal=True) == "Male" else 0

    st.subheader("History")
    fever_days = st.slider("Duration of fever (days)", 0, 30, 12)
    temp_max_c = st.slider("Max temperature (C)", 36.0, 42.0, 39.6, 0.1)

    st.subheader("Symptoms")
    symptoms = {}
    cols = st.columns(2)
    for i, s in enumerate(cfg.BINARY_SYMPTOMS):
        default = s in ("joint_pain", "headache", "rash", "retro_orbital_pain")
        symptoms[s] = int(cols[i % 2].checkbox(s.replace("_", " ").title(), value=default))

    st.subheader("Laboratory")
    platelet = st.slider("Platelets (x1000/uL)", 5, 700, 82)
    wbc = st.slider("WBC (x1000/uL)", 0.5, 40.0, 3.2, 0.1)
    hb = st.slider("Haemoglobin (g/dL)", 4.0, 19.0, 14.2, 0.1)
    crp = st.slider("CRP (mg/L)", 0, 350, 26)
    spo2 = st.slider("SpO2 (%)", 60, 100, 97)

patient = blank_patient() | {
    "age": float(age), "gender": gender, "region": region, "month": month,
    "fever_days": float(fever_days), "temp_max_c": float(temp_max_c),
    "platelet_count_k": float(platelet), "wbc_count_k": float(wbc),
    "hemoglobin_gdl": float(hb), "crp_mgl": float(crp), "spo2_pct": float(spo2),
} | symptoms

# ---------------------------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("Ranked differential diagnosis")
    out = predict(model, patient, top_k=cfg.NUM_CLASSES)
    top = out.iloc[0]
    st.metric("Most probable diagnosis", top["differential_diagnosis"],
              delta=top["confidence"])
    st.bar_chart(out.set_index("differential_diagnosis")["probability"])
    st.dataframe(out[["rank", "differential_diagnosis", "confidence"]],
                 hide_index=True, use_container_width=True)

with right:
    st.subheader("Same patient, different place")
    st.caption("Identical symptoms and labs — only the region and month change. "
               "This is the context-aware inference claim, made visible.")
    rows = []
    for r in cfg.REGIONS:
        for m, lbl in ((8, "Aug / Monsoon"), (1, "Jan / Winter")):
            alt = predict(model, patient | {"region": r, "month": m}, top_k=1)
            rows.append({"Region": r.replace("_", " "), "When": lbl,
                         "Top diagnosis": alt.iloc[0]["differential_diagnosis"],
                         "Confidence": alt.iloc[0]["confidence"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.divider()
st.info("**Decision support only.** This case now enters the "
        f"{cfg.QUARANTINE_DAYS}-day quarantine buffer and will not be used for "
        "training until the diagnosis is confirmed by laboratory assay.")
