import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import tensorflow as tf

# -----------------------------
# Basic environment setup
# -----------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# -----------------------------
# Artifact paths
# -----------------------------
ART = Path("artifacts")
MODEL_KERAS = ART / "aba_success_model.keras"
PIPE_PATH   = ART / "preprocessor.joblib"
INTV_PATH   = ART / "interventions.txt"

# -----------------------------
# Streamlit UI setup
# -----------------------------
st.set_page_config(page_title="SMART ABA — Your At-Home Therapist", layout="wide")
st.title("SMART ABA — Your At-Home Therapist")

# Load artifacts
# -----------------------------
assert MODEL_KERAS.exists(), f"Missing model at {MODEL_KERAS}"
assert PIPE_PATH.exists(),   f"Missing preprocessor at {PIPE_PATH}"
assert INTV_PATH.exists(),   f"Missing interventions at {INTV_PATH}"

model   = tf.keras.models.load_model(str(MODEL_KERAS))
preproc = joblib.load(PIPE_PATH)
with open(INTV_PATH) as f:
    ALL_INTERVENTIONS = [x.strip() for x in f if x.strip()]

# Get features from preprocessor
cat_features = preproc.transformers_[0][2]
num_features = preproc.transformers_[1][2]

# ==========================================================
# CONFIG: rename labels, include/exclude features, options
# ==========================================================
CONFIG = {
    "rename": {
        "parent_training_level": "Parent Training Level",
        "diagnosis_level": "Diagnosis Level",
        "behavior_type": "Behavior Type",
        "environment_noise_level": "Environment Noise",
        "reward_preference": "Reward Preference",
        "session_hour": "Session Hour (24h)",
        "behavior_intensity": "Behavior Intensity",
        "behavior_duration_min": "Behavior Duration (min)",
        "behavior_frequency_last_week": "Behavior Frequency (last week)",
        "therapy_hours_week": "Therapy Hours / Week",
    },
    "include": {
        "categorical": [
            "gender", "diagnosis_level", "setting", "antecedent", "behavior_type",
            "reinforcement", "parent_training_level", "medication_use", "school_support",
            "environment_noise_level", "reward_preference"
        ],
        "numeric": [
            "age", "behavior_intensity", "behavior_duration_min", "sibling_count",
            "therapy_hours_week", "behavior_frequency_last_week", "parent_stress_level",
            "session_hour"
        ]
    },
    "options": {
        "gender": ["Male", "Female", "Nonbinary", "Unknown"],
        "diagnosis_level": ["Level_1", "Level_2", "Level_3"],
        "setting": ["Home", "School", "Community", "Clinic"],
        "antecedent": ["Task_demand", "Transition", "Denied_access", "Attention", "Unknown"],
        "behavior_type": ["Non_compliance", "Tantrum", "Aggression", "SIB", "Elopement", "Other"],
        "reinforcement": ["Praise", "Token", "Edible", "Break", "Tangible", "Other"],
        "parent_training_level": ["Low", "Medium", "High", "Unknown"],
        "medication_use": ["Yes", "No", "Unknown"],
        "school_support": ["Yes", "No", "Unknown"],
        "environment_noise_level": ["Low", "Medium", "High"],
        "reward_preference": ["Edible", "Play", "Screen", "Social", "Other"],
    },
    "defaults": {
        "age": 8,
        "sibling_count": 1,
        "parent_stress_level": 4,
        "therapy_hours_week": 5,
        "behavior_intensity": 2,
        "behavior_duration_min": 5,
        "behavior_frequency_last_week": 7,
        "session_hour": 16
    },
    "enable_other_option": True,
}

def pretty(col: str) -> str:
    return CONFIG["rename"].get(col, col.replace("_", " ").title())

# Only use features your model expects
cat_features = [c for c in CONFIG["include"]["categorical"] if c in cat_features]
num_features = [c for c in CONFIG["include"]["numeric"] if c in num_features]

# -----------------------------
# Context input UI
# -----------------------------
st.subheader("Fill in the fields below to get your personalized Top-3 intervention recommendations.")
context = {}
cols = st.columns(3)
i = 0

# ---- categorical features ----
for c in cat_features:
    with cols[i % 3]:
        opts = CONFIG["options"].get(c, ["Unknown"])
        if CONFIG["enable_other_option"]:
            opts = opts + ["Other…"]
            choice = st.selectbox(pretty(c), opts, index=0)
            if choice == "Other…":
                custom = st.text_input(f"Enter custom value for {pretty(c)}")
                context[c] = custom.strip() if custom.strip() else opts[0]
            else:
                context[c] = choice
        else:
            context[c] = st.selectbox(pretty(c), opts, index=0)
    i += 1

# ---- numeric features ----
def default_num(col: str) -> float:
    return float(CONFIG["defaults"].get(col, 0.0))

for c in num_features:
    with cols[i % 3]:
        context[c] = st.number_input(pretty(c), value=default_num(c))
    i += 1

# -----------------------------
# Predict top-3 interventions
# -----------------------------
if st.button("Recommend Top-3 Interventions"):
    rows = []
    for itv in ALL_INTERVENTIONS:
        row = dict(context)
        row["intervention"] = itv
        rows.append(row)

    cand_df = pd.DataFrame(rows)

    # Ensure numeric columns exist & are numeric
    for c in num_features:
        if c not in cand_df.columns:
            cand_df[c] = 0.0
        cand_df[c] = pd.to_numeric(cand_df[c], errors="coerce").fillna(0.0)

    X_cand = preproc.transform(cand_df).astype("float32")
    probs = model.predict(X_cand, verbose=0).ravel()

    out = pd.DataFrame({
        "Intervention": cand_df["intervention"],
        "Predicted Success": probs
    }).sort_values("Predicted Success", ascending=False).head(3).reset_index(drop=True)

    st.success("🏆 Top-3 Recommendations")
    st.dataframe(out, use_container_width=True)
