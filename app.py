
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

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

    # Reduce lookalike collisions so the demo is clearer
    mask = (unknown["RT"].between(3.10, 3.40)) & (unknown["UV_max"].between(270, 276))
    unknown.loc[mask, "UV_max"] = unknown.loc[mask, "UV_max"] + rng.uniform(10, 40, mask.sum())

    return pd.concat([caffeine, unknown], ignore_index=True).sample(frac=1, random_state=seed)

@st.cache_data
def train():
    data = generate_synthetic_hplc_data()
    X = data.drop(columns=["Label"])
    y = data["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    # Anomaly detector trained on caffeine distribution only
    caffeine_train = X_train[y_train == "Caffeine"]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(caffeine_train)

    # Global importance for explanation
    sample_X = X_test.sample(min(300, len(X_test)), random_state=42)
    sample_y = y_test.loc[sample_X.index]
    perm = permutation_importance(clf, sample_X, sample_y, n_repeats=5, random_state=42)
    feat_importance = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

    return clf, iso, feat_importance

clf, iso, feat_importance = train()

st.title("ChromAI Demo (Synthetic Data)")
st.caption("Prototype: classify Caffeine vs Unknown using HPLC-UV features + anomaly flag.")

with st.expander("Project Workflow (Our Whole Process)", expanded=True):
    st.markdown("""
**1) Sample → Chromatography**
- Run HPLC (Phase 1: HPLC-UV for caffeine)
- Obtain chromatogram and peak metrics

**2) Feature Extraction**
We use: RT, peak area/height, width, tailing factor, asymmetry, UV wavelength.

**3) Machine Learning**
- Random Forest predicts the most likely compound class (Caffeine vs Unknown)
- Output includes confidence score (probability)

**4) Quality & Safety Layer**
- Isolation Forest flags abnormal peaks (distortion/interference/drift)

**5) Next Step (After Demo)**
- Replace synthetic data with real caffeine HPLC data from our chemist
- Expand from 1 compound → multiple compounds
- Later upgrade to LC-MS/MS features for clinical-level specificity
""")

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

    st.subheader("Why this prediction?")
    prob_dict = {classes[i]: float(proba[i]) for i in range(len(classes))}
    st.write("Class probabilities:", prob_dict)

    st.markdown("**Feature checks (caffeine-like ranges used in this demo):**")
    checks = [
        ("RT (min)", rt, "3.05–3.45", 3.05 <= rt <= 3.45),
        ("UV λmax (nm)", uv, "268–278", 268 <= uv <= 278),
        ("Width", width, "0.12–0.35", 0.12 <= width <= 0.35),
        ("Tailing", tail, "1.00–1.60", 1.00 <= tail <= 1.60),
        ("Asymmetry", asym, "0.85–1.40", 0.85 <= asym <= 1.40),
        ("Area_norm", area, "0.30–1.20", 0.30 <= area <= 1.20),
    ]
    st.dataframe(pd.DataFrame(checks, columns=["Feature", "Your value", "Expected range", "Within range?"]),
                 use_container_width=True)

    st.markdown("**Model learned importance (global):**")
    st.bar_chart(feat_importance)

    if pred == "Unknown":
        st.error(
            "Predicted **Unknown** because one or more inputs are outside the caffeine-like pattern "
            "the model learned (or the probability difference is small). Check the table above — "
            "features with 'Within range? = False' are the main reasons."
        )
    else:
        st.info(
            "Predicted **Caffeine** because your input features match the caffeine-like pattern "
            "in retention time, UV behavior, and peak shape."
        )
