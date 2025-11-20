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

CSV_PATH = r"D:\SDSU\SEM THREE\BDA 696\ABA\data\ABA_Sessions_with_TimeAndDay.csv"   # put this file in the same folder as this script
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

# scikit-learn ≥1.4 uses `sparse_output`; older versions used `sparse`
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", ohe, categorical_features),
        ("num", StandardScaler(), numeric_features),
    ],
    remainder="drop",
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
# =========================
# 7A) Add baselines: Logistic Regression & XGBoost
# =========================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("WARNING: xgboost not installed; skipping XGBClassifier.")

# Helper: readable feature names from preprocessor
def get_feature_names(preprocessor, cat_features, num_features):
    cat_names = []
    if "cat" in preprocessor.named_transformers_:
        ohe = preprocessor.named_transformers_["cat"]
        cat_names = ohe.get_feature_names_out(cat_features).tolist()
    return cat_names + list(num_features)

feature_names = get_feature_names(preprocessor, categorical_features, numeric_features)
print("Encoded feature count:", len(feature_names))

# Helper: evaluate metrics
def evaluate_all(y_true, y_proba, threshold=0.5, label="model"):
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "Model": label,
        "ROC AUC": roc_auc_score(y_true, y_proba),
        "PR AUC": average_precision_score(y_true, y_proba),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics, y_pred

# Helper: plot confusion matrix
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["0","1"]); ax.set_yticklabels(["0","1"])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# --- Keras probabilities on test (already trained) ---
keras_test_proba = model.predict(X_test_np, verbose=0).ravel()
keras_metrics, keras_pred = evaluate_all(y_test, keras_test_proba, threshold=0.5, label="Keras (DNN)")
plot_cm(y_test, keras_pred, "Confusion Matrix — Keras")

# --- Logistic Regression (with class_weight to handle imbalance) ---
lr = LogisticRegression(
    penalty="l2", C=1.0, solver="lbfgs",
    max_iter=2000, n_jobs=-1, class_weight="balanced"
)
lr.fit(X_train_np, y_train)
lr_val_proba  = lr.predict_proba(X_val_np)[:,1]
lr_test_proba = lr.predict_proba(X_test_np)[:,1]
lr_metrics, lr_pred = evaluate_all(y_test, lr_test_proba, threshold=0.5, label="Logistic Regression")
plot_cm(y_test, lr_pred, "Confusion Matrix — Logistic Regression")

# --- XGBoost (if available) ---
if HAS_XGB:
    # Compute scale_pos_weight ~ (neg/pos) using train set
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = max((neg / max(pos,1)), 1.0)

    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
        scale_pos_weight=spw,
        n_jobs=-1
    )
    xgb.fit(X_train_np, y_train, eval_set=[(X_val_np, y_val)], verbose=False)
    xgb_test_proba = xgb.predict_proba(X_test_np)[:,1]
    xgb_metrics, xgb_pred = evaluate_all(y_test, xgb_test_proba, threshold=0.5, label="XGBoost")
    plot_cm(y_test, xgb_pred, "Confusion Matrix — XGBoost")
else:
    xgb_metrics = None

# --- Compare metrics ---
import pandas as pd
metrics_list = [keras_metrics, lr_metrics] + ([xgb_metrics] if xgb_metrics else [])
metrics_df = pd.DataFrame(metrics_list)
print("\n=== Test Metrics Comparison ===")
print(metrics_df.to_string(index=False))

# =========================
# 7B) SHAP analysis
# =========================
import shap
shap.initjs()

# Choose a small background set for KernelExplainer
bg_n = min(200, X_train_np.shape[0])
bg = shap.sample(X_train_np, bg_n, random_state=42)

# 1) SHAP for XGB (fast & accurate for trees)
if HAS_XGB:
    expl_xgb = shap.TreeExplainer(xgb, feature_perturbation="tree_path_dependent")
    X_shap = X_test_np[: min(200, X_test_np.shape[0])]
    shap_vals_xgb = expl_xgb.shap_values(X_shap)
    plt.figure()
    shap.summary_plot(shap_vals_xgb, X_shap, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Global Importance — XGBoost")
    plt.tight_layout(); plt.show()

    plt.figure()
    shap.summary_plot(shap_vals_xgb, X_shap, feature_names=feature_names, show=False)
    plt.title("SHAP Beeswarm — XGBoost")
    plt.tight_layout(); plt.show()

# 2) SHAP for Keras & Logistic Regression (KernelExplainer)
def predict_keras(X):
    return model.predict(X, verbose=0).ravel()

def predict_lr(X):
    return lr.predict_proba(X)[:,1]

# Keras
expl_ker = shap.KernelExplainer(predict_keras, bg)
X_shap_small = X_test_np[: min(120, X_test_np.shape[0])]
shap_vals_ker = expl_ker.shap_values(X_shap_small, nsamples="auto")
plt.figure()
shap.summary_plot(shap_vals_ker, X_shap_small, feature_names=feature_names, plot_type="bar", show=False)
plt.title("SHAP Global Importance — Keras")
plt.tight_layout(); plt.show()

# Logistic Regression
expl_lr = shap.KernelExplainer(predict_lr, bg)
shap_vals_lr = expl_lr.shap_values(X_shap_small, nsamples="auto")
plt.figure()
shap.summary_plot(shap_vals_lr, X_shap_small, feature_names=feature_names, plot_type="bar", show=False)
plt.title("SHAP Global Importance — Logistic Regression")
plt.tight_layout(); plt.show()

# Optional: local waterfall for one test instance (Keras)
instance = X_test_np[0:1]
sv_single = expl_ker.shap_values(instance, nsamples="auto")[0]
base_val = expl_ker.expected_value
try:
    exp_single = shap.Explanation(
        values=sv_single,
        base_values=np.array([base_val]),
        data=instance[0],
        feature_names=feature_names
    )
    plt.figure()
    shap.plots.waterfall(exp_single, max_display=15, show=False)
    plt.title("SHAP Waterfall — Keras (one test case)")
    plt.tight_layout(); plt.show()
except Exception:
    order = np.argsort(np.abs(sv_single))[-15:][::-1]
    plt.figure(figsize=(6,5))
    plt.barh([feature_names[i] for i in order][::-1], np.array(sv_single)[order][::-1])
    plt.title("Top-15 SHAP contributions — Keras (fallback)")
    plt.tight_layout(); plt.show()

# =========================
# 7C) LIME (local explanation for same instance)
# =========================
from lime.lime_tabular import LimeTabularExplainer

lime_explainer = LimeTabularExplainer(
    training_data=bg,  # background is fine
    feature_names=feature_names,
    discretize_continuous=True,
    mode="classification",
    class_names=["fail","success"]
)

# Use Keras for the local explanation; you can swap to XGB by changing predict_fn
lime_exp = lime_explainer.explain_instance(
    data_row=instance[0],
    predict_fn=lambda X: np.vstack([1 - predict_keras(X), predict_keras(X)]).T,
    num_features=10
)
fig = lime_exp.as_pyplot_figure()
plt.title("LIME Local Explanation — Keras (one test case)")
plt.tight_layout(); plt.show()

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