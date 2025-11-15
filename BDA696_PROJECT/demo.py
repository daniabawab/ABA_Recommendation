import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

CSV_PATH = "/Users/daniabawab/Desktop/BDA696_PROJECT/data/ABA_Sessions_with_TimeAndDay.csv"   # put this file in the same folder as this script
df = pd.read_csv(CSV_PATH)

print("Data shape:", df.shape)
print("Columns:", list(df.columns))

target_col = "outcome_success"

cat_cols = [
    "gender",
    "diagnosis_level",
    "setting",
    "antecedent",
    "behavior_type",
    "reinforcement",
    "parent_training_level",
    "medication_use",
    "school_support",
    "environment_noise_level",
    "reward_preference",
    "intervention",          # <- IMPORTANT: used at train time AND at inference (we will try each)
]

num_cols = [
    "age",
    "behavior_intensity",
    "behavior_duration_min",
    "sibling_count",
    "therapy_hours_week",
    "behavior_frequency_last_week",
    "parent_stress_level",
    "session_hour",
]
# Drop columns we won’t use (ok if some don’t exist)
drop_cols = [
    "day_of_week", "Description", "timestamp", "session_time", "event_id"
]
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=[c])

# Keep only needed columns + target
keep_cols = [c for c in cat_cols + num_cols + [target_col] if c in df.columns]
df = df[keep_cols].copy()

# Basic cleaning: drop rows with missing target; (optionally) fill/drop feature NaNs
df = df.dropna(subset=[target_col])
# Simple imputation strategy for beginners: fill numeric NaNs with median, categorical with "Unknown"
for c in num_cols:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].fillna("Unknown")

# Ensure binary target is numeric 0/1
df[target_col] = df[target_col].astype(int)

# Record full set of unique interventions for inference-time search
ALL_INTERVENTIONS = sorted(df["intervention"].dropna().unique().tolist())
print("Interventions ({}):".format(len(ALL_INTERVENTIONS)), ALL_INTERVENTIONS[:10], "..." if len(ALL_INTERVENTIONS) > 10 else "")

# -------------------------
# 3) TRAIN/VAL/TEST SPLIT
# -------------------------
X = df.drop(columns=[target_col])
y = df[target_col].values

# Train/val/test = 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
print("Train/Val/Test sizes:", len(X_train), len(X_val), len(X_test))

# -------------------------
# 4) PREPROCESSING PIPELINE
# -------------------------
# Build a ColumnTransformer that one-hot encodes categoricals and scales numerics
categorical_features = [c for c in cat_cols if c in X.columns]
numeric_features = [c for c in num_cols if c in X.columns]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ],
    remainder="drop"
)

# Fit on train, transform all splits to Numpy arrays
preprocessor.fit(X_train)
X_train_np = preprocessor.transform(X_train)
X_val_np = preprocessor.transform(X_val)
X_test_np = preprocessor.transform(X_test)

input_dim = X_train_np.shape[1]
print("Input dimension after encoding:", input_dim)

# -------------------------
# 5) BUILD A SMALL KERAS MODEL
# -------------------------
def make_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")  # probability of success
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"]
    )
    return model

model = make_model(input_dim)
model.summary()

# -------------------------
# 6) TRAIN
# -------------------------
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc", mode="max"),
    # Optional: save checkpoints
    # keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_auc", mode="max")
]

history = model.fit(
    X_train_np, y_train,
    validation_data=(X_val_np, y_val),
    epochs=50,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)

# -------------------------
# 7) EVALUATE
# -------------------------
test_loss, test_auc, test_acc = model.evaluate(X_test_np, y_test, verbose=0)
print(f"Test AUC: {test_auc:.3f} | Test Acc: {test_acc:.3f}")

# -------------------------
# 8) SAVE MODEL + PIPELINE
# -------------------------
Path("artifacts").mkdir(exist_ok=True)
MODEL_PATH = "artifacts/aba_success_model.keras"
PIPE_PATH = "artifacts/preprocessor.joblib"
INTV_PATH = "artifacts/interventions.txt"

model.save(MODEL_PATH)
joblib.dump(preprocessor, PIPE_PATH)
with open(INTV_PATH, "w") as f:
    for itv in ALL_INTERVENTIONS:
        f.write(itv + "\n")

print("Saved:", MODEL_PATH, PIPE_PATH, INTV_PATH)

# -------------------------
# 9) RECOMMENDER FUNCTION
#    Given context (WITHOUT day_of_week), try each intervention and return top-k
# -------------------------
def load_artifacts():
    mdl = keras.models.load_model(MODEL_PATH)
    pp = joblib.load(PIPE_PATH)
    with open(INTV_PATH, "r") as f:
        intvs = [line.strip() for line in f if line.strip()]
    return mdl, pp, intvs

def recommend(context: dict, top_k: int = 3):
    """
    context: dict with keys matching the feature columns EXCEPT 'intervention'.
             e.g. {
               "gender": "Male",
               "diagnosis_level": "Level_2",
               "setting": "Home",
               "antecedent": "Task_demand",
               "behavior_type": "Non_compliance",
               "reinforcement": "Praise",
               "parent_training_level": "Medium",
               "medication_use": "No",
               "school_support": "Yes",
               "environment_noise_level": "Low",
               "reward_preference": "Edible",
               "age": 8,
               "behavior_intensity": 2,
               "behavior_duration_min": 6,
               "sibling_count": 1,
               "therapy_hours_week": 5,
               "behavior_frequency_last_week": 7,
               "parent_stress_level": 4,
               "session_hour": 16
             }
    """
    mdl, pp, intvs = load_artifacts()

    # Build a small DataFrame: one row per intervention
    rows = []
    for itv in intvs:
        row = context.copy()
        row["intervention"] = itv
        # Fill missing fields if user provided fewer keys
        for c in categorical_features:
            row.setdefault(c, "Unknown")
        for c in numeric_features:
            row.setdefault(c, float(np.nan))
        rows.append(row)

    cand_df = pd.DataFrame(rows)

    # Simple numeric imputation at inference time (median from training set would be better; using fit data is fine)
    for c in numeric_features:
        if cand_df[c].isna().any():
            # fall back to train median if available, else 0
            median = X_train[c].median() if c in X_train.columns else 0.0
            cand_df[c] = cand_df[c].fillna(median)

    X_cand = pp.transform(cand_df)
    preds = mdl.predict(X_cand, verbose=0).ravel()  # success probabilities

    cand_df["pred_success"] = preds
    cand_df = cand_df.sort_values("pred_success", ascending=False)

    top = cand_df[["intervention", "pred_success"]].head(top_k)
    return top.reset_index(drop=True)

# -------------------------
# 10) QUICK DEMO
# -------------------------
# Build a demo context from one row in the dataset (we'll remove intervention & target and exclude day_of_week)
sample = X_test.sample(1, random_state=7).iloc[0].to_dict()
# Remove 'intervention' from context; we will search over all interventions
if "intervention" in sample:
    sample.pop("intervention")

print("\nDEMO CONTEXT (no day_of_week):")
for k, v in sample.items():
    print(f"  {k}: {v}")

top3 = recommend(sample, top_k=3)
print("\nTOP-3 RECOMMENDED INTERVENTIONS (highest predicted success):")
print(top3.to_string(index=False))