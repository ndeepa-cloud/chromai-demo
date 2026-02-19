import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="ChromAI Demo", layout="centered")

def generate_synthetic_hplc_data(n_caffeine=800, n_unknown=800, seed=42):
    rng = np.random.default_rng(seed)

    caffeine = pd.DataFrame({
        "RT": rng.normal(3.25, 0.06, n_caffeine).clip(3.05, 3.45),
        "Area_norm": rng.normal(0.85, 0.10, n_caffeine).clip(0.30, 1.20),
        "Width": rng.normal(0.22, 0.03, n_caffeine).clip(0.12, 0.35),
        "Tailing": rng.normal(1.15, 0.10, n_caffeine).clip(1.00, 1.60),
        "Asymmetry": rng.normal(1.05, 0.08, n_caffeine).clip(0.85, 1.40),
        "UV_max": rng.normal(273, 1.5, n_caffeine).clip(268, 278),
        "Label": "Caffeine"
    })

    unknown = pd.DataFrame({
        "RT": rng.uniform(2.2, 5.0, n_unknown),
        "Area_norm": rng.uniform(0.05, 1.40, n_unknown),
        "Width": rng.uniform(0.10, 0.60, n_unknown),
        "Tailing": rng.uniform(1.00, 2.50, n_unknown),
        "Asymmetry": rng.uniform(0.70, 2.20, n_unknown),
        "UV_max": rng.uniform(210, 330, n_unknown),
        "Label": "Unknown"
    })

    # Reduce lookalike collisions (so demo is clearer)
    mask = (unknown["RT"].between(3.10, 3.40)) & (unknown["UV_max"].between(270, 276))
    unknown.loc[mask, "UV_max"] = unknown.loc[mask, "UV_max"] + rng.uniform(10, 40, mask.sum())

    return pd.concat([caffeine, unknown], ignore_index=True).sample(frac=1, random_state=seed)

@st.cache_data
def train():
    data = generate_synthetic_hplc_data()
    X = data.drop(columns=["Label"])
    y = data["Label"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    caffeine_train = X_train[y_train == "Caffeine"]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(caffeine_train)

    return clf, iso

clf, iso = train()

st.title("ChromAI Demo (Synthetic Data)")
st.caption("Prototype: classify Caffeine vs Unknown using HPLC-UV features + anomaly flag.")

st.subheader("Enter Peak Features")
rt = st.number_input("Retention Time (min)", value=3.27, step=0.01, format="%.2f")
area = st.number_input("Normalized Peak Area", value=0.90, step=0.01, format="%.2f")
width = st.number_input("Peak Width", value=0.21, step=0.01, format="%.2f")
tail = st.number_input("Tailing Factor", value=1.12, step=0.01, format="%.2f")
asym = st.number_input("Asymmetry", value=1.04, step=0.01, format="%.2f")
uv = st.number_input("UV λmax (nm)", value=273, step=1)

if st.button("Predict"):
    sample = pd.DataFrame([{
        "RT": rt, "Area_norm": area, "Width": width,
        "Tailing": tail, "Asymmetry": asym, "UV_max": uv
    }])

    proba = clf.predict_proba(sample)[0]
    classes = clf.classes_
    best_idx = int(np.argmax(proba))
    pred = classes[best_idx]
    conf = float(proba[best_idx])

    anomaly = (iso.predict(sample)[0] == -1)

    st.success(f"Prediction: {pred}  |  Confidence: {conf*100:.1f}%")
    st.warning("Peak Quality Flag: ANOMALY (review suggested)") if anomaly else st.info("Peak Quality Flag: OK")

