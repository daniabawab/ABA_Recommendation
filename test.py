# ====== FAST, RELIABLE FULL PIPELINE ======
# - Trains Keras, Logistic Regression, XGBoost (if available)
# - Compares metrics
# - Confusion matrices
# - Permutation importance (fast: defaults to LR)
# - Saves artifacts + Top-3 recommender
# =========================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # stable numerics on CPU

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------- Reproducibility --------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# -------- Data --------
CSV_PATH = r"D:\SDSU\SEM THREE\BDA 696\ABA\data\ABA_Sessions_with_TimeAndDay.csv"
df = pd.read_csv(CSV_PATH)

print("Data shape:", df.shape)
print("Columns:", list(df.columns))

target_col = "outcome_success"
cat_cols = [
    "gender","diagnosis_level","setting","antecedent","behavior_type",
    "reinforcement","parent_training_level","medication_use","school_support",
    "environment_noise_level","reward_preference","intervention"
]
num_cols = [
    "age","behavior_intensity","behavior_duration_min","sibling_count",
    "therapy_hours_week","behavior_frequency_last_week","parent_stress_level","session_hour",
]
drop_cols = ["day_of_week","Description","timestamp","session_time","event_id"]
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=[c])

keep_cols = [c for c in cat_cols + num_cols + [target_col] if c in df.columns]
df = df[keep_cols].copy()

# basic cleaning
df = df.dropna(subset=[target_col])
for c in num_cols:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].fillna("Unknown")
df[target_col] = df[target_col].astype(int)

ALL_INTERVENTIONS = sorted(df["intervention"].dropna().unique().tolist())
print(f"Interventions ({len(ALL_INTERVENTIONS)}):", ALL_INTERVENTIONS[:10], "" if len(ALL_INTERVENTIONS) <= 10 else "...")

# -------- Split --------
X = df.drop(columns=[target_col])
y = df[target_col].values
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)
print("Train/Val/Test sizes:", len(X_train), len(X_val), len(X_test))

categorical_features = [c for c in cat_cols if c in X.columns]
numeric_features = [c for c in num_cols if c in X.columns]

# OneHotEncoder compatibility
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

preprocessor.fit(X_train)
X_train_np = preprocessor.transform(X_train)
X_val_np   = preprocessor.transform(X_val)
X_test_np  = preprocessor.transform(X_test)

def get_feature_names(preprocessor, cat_features, num_features):
    names = []
    if hasattr(preprocessor, "named_transformers_") and "cat" in preprocessor.named_transformers_:
        enc = preprocessor.named_transformers_["cat"]
        if hasattr(enc, "get_feature_names_out"):
            names = enc.get_feature_names_out(cat_features).tolist()
        else:
            names = enc.get_feature_names(cat_features).tolist()
    return names + list(num_features)

feature_names = get_feature_names(preprocessor, categorical_features, numeric_features)
input_dim = X_train_np.shape[1]
print("Input dimension after encoding:", input_dim)

# -------- Keras model --------
def make_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"]
    )
    return model

model = make_model(input_dim)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True, monitor="val_auc", mode="max"
    )
]
history = model.fit(
    X_train_np, y_train,
    validation_data=(X_val_np, y_val),
    epochs=30,                # fewer epochs -> quicker results (EarlyStopping will cut earlier)
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)

# -------- Evaluate Keras --------
test_loss, test_auc, test_acc = model.evaluate(X_test_np, y_test, verbose=0)
print(f"Test AUC: {test_auc:.3f} | Test Acc: {test_acc:.3f}")

# -------- Baselines: LR and XGB --------
def evaluate_all(y_true, y_proba, threshold=0.5, label="model"):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "Model": label,
        "ROC AUC": roc_auc_score(y_true, y_proba),
        "PR AUC": average_precision_score(y_true, y_proba),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }, y_pred

# Keras probs
keras_test_proba = model.predict(X_test_np, verbose=0).ravel()
keras_metrics, keras_pred = evaluate_all(y_test, keras_test_proba, 0.5, "Keras (DNN)")

# Logistic Regression (fast, strong baseline)
lr = LogisticRegression(
    penalty="l2", C=1.0, solver="lbfgs",
    max_iter=2000, n_jobs=-1, class_weight="balanced"
)
lr.fit(X_train_np, y_train)
lr_test_proba = lr.predict_proba(X_test_np)[:,1]
lr_metrics, lr_pred = evaluate_all(y_test, lr_test_proba, 0.5, "Logistic Regression")

# XGBoost (optional)
try:
    from xgboost import XGBClassifier
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = max((neg / max(pos, 1)), 1.0)

    xgb = XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="binary:logistic", tree_method="hist",
        random_state=RANDOM_STATE, scale_pos_weight=spw, n_jobs=-1
    )
    xgb.fit(X_train_np, y_train, eval_set=[(X_val_np, y_val)], verbose=False)
    xgb_test_proba = xgb.predict_proba(X_test_np)[:,1]
    xgb_metrics, xgb_pred = evaluate_all(y_test, xgb_test_proba, 0.5, "XGBoost")
    HAS_XGB = True
except Exception as e:
    print("XGBoost not available -> skipping. Details:", e)
    xgb_metrics, xgb_pred, HAS_XGB = None, None, False

# -------- Compare metrics --------
metrics_list = [keras_metrics, lr_metrics] + ([xgb_metrics] if HAS_XGB else [])
metrics_df = pd.DataFrame(metrics_list)
print("\n=== Test Metrics Comparison ===")
print(metrics_df.to_string(index=False))

# -------- Confusion matrices --------
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

plot_cm(y_test, keras_pred, "Confusion Matrix — Keras")
plot_cm(y_test, lr_pred,    "Confusion Matrix — Logistic Regression")
if HAS_XGB:
    plot_cm(y_test, xgb_pred,   "Confusion Matrix — XGBoost")

# -------- Permutation Importance (fast: use LR by default) --------
class KerasWrapper(BaseEstimator):
    """Make Keras look like a proper sklearn classifier."""
    def __init__(self, keras_model, threshold=0.5):
        self.model = keras_model
        self.threshold = threshold
        self._estimator_type = "classifier"
        self.classes_ = np.array([0,1])
    def fit(self, X, y=None): return self
    def predict_proba(self, X):
        p = self.model.predict(X, verbose=0).ravel()
        return np.column_stack([1.0 - p, p])
    def predict(self, X):
        p = self.model.predict(X, verbose=0).ravel()
        return (p >= self.threshold).astype(int)

def aggregate_ohe(df_feat_imp: pd.DataFrame, categorical_features, numeric_features):
    rows, used = [], set()
    for col in categorical_features:
        prefix = f"{col}_"
        mask = df_feat_imp["feature"].str.startswith(prefix)
        if mask.any():
            rows.append({"feature": col, "importance": float(df_feat_imp.loc[mask, "importance"].sum())})
            used.update(df_feat_imp.loc[mask, "feature"].tolist())
    for col in numeric_features:
        if (df_feat_imp["feature"] == col).any():
            val = float(df_feat_imp.loc[df_feat_imp["feature"] == col, "importance"].values[0])
            rows.append({"feature": col, "importance": val})
    leftovers = df_feat_imp[~df_feat_imp["feature"].isin(used | set(numeric_features))]
    if not leftovers.empty:
        rows += leftovers.to_dict("records")
    agg = (pd.DataFrame(rows)
           .groupby("feature", as_index=False)["importance"].sum()
           .sort_values("importance", ascending=False)
           .reset_index(drop=True))
    return agg

def get_feature_names(preprocessor, cat_features, num_features):
    names = []
    if hasattr(preprocessor, "named_transformers_") and "cat" in preprocessor.named_transformers_:
        enc = preprocessor.named_transformers_["cat"]
        if hasattr(enc, "get_feature_names_out"):
            names = enc.get_feature_names_out(cat_features).tolist()
        else:
            names = enc.get_feature_names(cat_features).tolist()
    return names + list(num_features)

feature_names = get_feature_names(preprocessor, categorical_features, numeric_features)

# Choose model for PI (LR is much faster than Keras; switch to KerasWrapper(model) if needed)
est_for_pi = lr
n_jobs_pi = max(1, min(os.cpu_count() or 1, 8))
perm = permutation_importance(
    estimator=est_for_pi,
    X=X_test_np, y=y_test,
    scoring="roc_auc",
    n_repeats=5,            # keep small for speed
    random_state=RANDOM_STATE,
    n_jobs=n_jobs_pi
)
perm_df = pd.DataFrame({
    "feature": feature_names,
    "importance": perm.importances_mean,
    "std": perm.importances_std
}).sort_values("importance", ascending=False)
perm_agg = aggregate_ohe(perm_df[["feature","importance"]], categorical_features, numeric_features)

TOPK = 20
print("\n=== Permutation Importance (ROC AUC drop) — per dummy (Top {}): ===".format(TOPK))
print(perm_df.head(TOPK).to_string(index=False))
print("\n=== Permutation Importance (ROC AUC drop) — aggregated by original column (Top {}): ===".format(TOPK))
print(perm_agg.head(TOPK).to_string(index=False))

ART = Path("artifacts"); ART.mkdir(exist_ok=True, parents=True)
perm_df.to_csv(ART / "permutation_importance_per_dummy.csv", index=False)
perm_agg.to_csv(ART / "permutation_importance_aggregated.csv", index=False)
plt.figure(figsize=(8,6))
plt.barh(perm_agg["feature"].head(TOPK)[::-1], perm_agg["importance"].head(TOPK)[::-1])
plt.xlabel("Mean ROC AUC decrease when shuffled")
plt.title("Permutation Importance (Aggregated) — Top 20 (LR)")
plt.tight_layout()
plt.savefig(ART / "permutation_importance_aggregated_top20.png", dpi=150, bbox_inches="tight")
plt.show()

# -------- Save artifacts --------
MODEL_PATH = ART / "aba_success_model.keras"
PIPE_PATH  = ART / "preprocessor.joblib"
INTV_PATH  = ART / "interventions.txt"

model.save(MODEL_PATH)
joblib.dump(preprocessor, PIPE_PATH)
with open(INTV_PATH, "w") as f:
    for itv in ALL_INTERVENTIONS:
        f.write(itv + "\n")
print("Saved:", MODEL_PATH, PIPE_PATH, INTV_PATH)

# -------- Recommender --------
def load_artifacts():
    mdl = keras.models.load_model(MODEL_PATH)
    pp = joblib.load(PIPE_PATH)
    with open(INTV_PATH, "r") as f:
        intvs = [line.strip() for line in f if line.strip()]
    return mdl, pp, intvs

def recommend(context: dict, top_k: int = 3):
    mdl, pp, intvs = load_artifacts()
    rows = []
    for itv in intvs:
        row = context.copy()
        row["intervention"] = itv
        for c in categorical_features:
            row.setdefault(c, "Unknown")
        for c in numeric_features:
            row.setdefault(c, float(np.nan))
        rows.append(row)
    cand_df = pd.DataFrame(rows)
    for c in numeric_features:
        if cand_df[c].isna().any():
            median = X_train[c].median() if c in X_train.columns else 0.0
            cand_df[c] = cand_df[c].fillna(median)
    X_cand = pp.transform(cand_df)
    preds = mdl.predict(X_cand, verbose=0).ravel()
    cand_df["pred_success"] = preds
    cand_df = cand_df.sort_values("pred_success", ascending=False)
    top = cand_df[["intervention","pred_success"]].head(top_k)
    return top.reset_index(drop=True)

# -------- Quick demo --------
sample = X_test.sample(1, random_state=7).iloc[0].to_dict()
if "intervention" in sample:
    sample.pop("intervention")
print("\nDEMO CONTEXT (no day_of_week):")
for k, v in sample.items():
    print(f"  {k}: {v}")

top3 = recommend(sample, top_k=3)
print("\nTOP-3 RECOMMENDED INTERVENTIONS (highest predicted success):")
print(top3.to_string(index=False))
