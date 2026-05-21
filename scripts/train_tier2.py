"""
Train Tier-2: CatBoost multi-class classifier with unknown detection threshold.
Classifies: DoS, BruteForce, PortScan
Unknown: confidence < 0.65
"""

import os
import json
import sys
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from catboost import CatBoostClassifier, Pool
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

base = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base, "config", "config.json"), "r") as f:
    config = json.load(f)

PROCESSED_DIR = os.path.join(base, "data", "processed")
MODELS_DIR = os.path.join(base, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

UNKNOWN_THRESHOLD = config["tier2"]["unknown_threshold"]

logger.info("Loading Tier-2 data...")
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "tier2_X_train.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "tier2_X_test.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "tier2_y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "tier2_y_test.csv")).values.ravel()
feature_order = joblib.load(os.path.join(PROCESSED_DIR, "feature_order.joblib"))

X_zeroday = pd.read_csv(os.path.join(PROCESSED_DIR, "zeroday_X.csv"))
y_zeroday = pd.read_csv(os.path.join(PROCESSED_DIR, "zeroday_y.csv")).values.ravel()

logger.info("Tier-2 Train: %s, Test: %s", X_train.shape, X_test.shape)
logger.info("Zero-day eval: %s", X_zeroday.shape)
logger.info("Train class distribution:\n%s", pd.Series(y_train).value_counts().to_string())
logger.info("Test class distribution:\n%s", pd.Series(y_test).value_counts().to_string())
logger.info("Zero-day labels:\n%s", pd.Series(y_zeroday).value_counts().to_string())

logger.info("Preprocessing...")
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
X_zeroday = X_zeroday.replace([np.inf, -np.inf], np.nan).fillna(0)

X_train = X_train[feature_order]
X_test = X_test[feature_order]
X_zeroday = X_zeroday[feature_order]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

logger.info("Label mapping:")
for i, cls in enumerate(label_encoder.classes_):
    logger.info("  %s -> %d", cls, i)

scaler_t2 = StandardScaler()
X_train_scaled = scaler_t2.fit_transform(X_train)
X_test_scaled = scaler_t2.transform(X_test)
X_zeroday_scaled = scaler_t2.transform(X_zeroday)

joblib.dump(scaler_t2, os.path.join(MODELS_DIR, "tier2_preprocessor.joblib"))
joblib.dump(label_encoder, os.path.join(MODELS_DIR, "tier2_label_encoder.joblib"))
joblib.dump(feature_order, os.path.join(MODELS_DIR, "tier2_feature_order.joblib"))

logger.info("Computing class weights...")
class_weights = compute_class_weight("balanced", classes=np.unique(y_train_encoded), y=y_train_encoded)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
logger.info("Class weights: %s", class_weight_dict)

logger.info("Training CatBoost...")
catboost_params = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3,
    "class_weights": class_weight_dict,
    "eval_metric": "TotalF1",
    "random_seed": 42,
    "verbose": 100,
    "early_stopping_rounds": 50,
}

cat_model = CatBoostClassifier(**catboost_params)

train_pool = Pool(X_train_scaled, y_train_encoded)
val_pool = Pool(X_test_scaled, y_test_encoded)

cat_model.fit(train_pool, eval_set=val_pool)

logger.info("Evaluating on test set...")
y_pred = cat_model.predict(X_test_scaled)
y_pred_proba = cat_model.predict_proba(X_test_scaled)
y_pred_labels = label_encoder.inverse_transform(y_pred.astype(int))
y_test_labels = label_encoder.inverse_transform(y_test_encoded)

logger.info("CatBoost Classification Report:")
logger.info("\n%s", classification_report(y_test_labels, y_pred_labels))

cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)
logger.info("Confusion Matrix:\n%s", pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_).to_string())

precision = precision_score(y_test_labels, y_pred_labels, average="weighted", zero_division=0)
recall = recall_score(y_test_labels, y_pred_labels, average="weighted", zero_division=0)
f1 = f1_score(y_test_labels, y_pred_labels, average="weighted", zero_division=0)
logger.info("Weighted Precision: %.4f", precision)
logger.info("Weighted Recall: %.4f", recall)
logger.info("Weighted F1: %.4f", f1)

logger.info("Evaluating unknown detection threshold...")
max_probas = np.max(y_pred_proba, axis=1)

logger.info("Confidence Distribution on Known Test Set:")
logger.info("  Mean: %.4f", max_probas.mean())
logger.info("  Std:  %.4f", max_probas.std())
logger.info("  Min:  %.4f", max_probas.min())
logger.info("  25%%:  %.4f", np.percentile(max_probas, 25))
logger.info("  50%%:  %.4f", np.percentile(max_probas, 50))
logger.info("  75%%:  %.4f", np.percentile(max_probas, 75))

unknown_count = (max_probas < UNKNOWN_THRESHOLD).sum()
total = len(max_probas)
logger.info("Known attacks below threshold %.2f: %d/%d (%.1f%%)", UNKNOWN_THRESHOLD, unknown_count, total, unknown_count / total * 100)

logger.info("Zero-Day Evaluation...")
zeroday_proba = cat_model.predict_proba(X_zeroday_scaled)
zeroday_max_proba = np.max(zeroday_proba, axis=1)

logger.info("  Mean confidence: %.4f", zeroday_max_proba.mean())
logger.info("  Min confidence: %.4f", zeroday_max_proba.min())
logger.info("  Max confidence: %.4f", zeroday_max_proba.max())

zeroday_unknown = (zeroday_max_proba < UNKNOWN_THRESHOLD).sum()
zeroday_total = len(zeroday_max_proba)
logger.info("  Unknown detections: %d/%d (%.1f%%)", zeroday_unknown, zeroday_total, zeroday_unknown / zeroday_total * 100)

for label in np.unique(y_zeroday):
    mask = y_zeroday == label
    if mask.sum() > 0:
        probs = zeroday_max_proba[mask]
        unknown_pct = (probs < UNKNOWN_THRESHOLD).sum() / len(probs) * 100
        logger.info("  %s: %.1f%% Unknown (mean conf: %.4f)", label, unknown_pct, probs.mean())

logger.info("End-to-End Pipeline Test...")
lgbm_model = joblib.load(os.path.join(MODELS_DIR, "tier1_lgbm.joblib"))
iforest = joblib.load(os.path.join(MODELS_DIR, "tier1_iforest.joblib"))
tier1_scaler = joblib.load(os.path.join(MODELS_DIR, "tier1_preprocessor.joblib"))
tier1_thresholds = joblib.load(os.path.join(MODELS_DIR, "tier1_thresholds.joblib"))

LGBM_THRESHOLD = tier1_thresholds["lgbm_threshold"]

X_zeroday_t1 = X_zeroday[feature_order]
X_zeroday_t1_scaled = tier1_scaler.transform(X_zeroday_t1)

lgbm_proba = lgbm_model.predict_proba(X_zeroday_t1)[:, 1]
if_pred = iforest.predict(X_zeroday_t1_scaled)
if_anomaly = if_pred == -1
lgbm_anomaly = lgbm_proba > LGBM_THRESHOLD
tier1_anomaly = lgbm_anomaly | if_anomaly

logger.info("Tier-1 on Zero-Day Data:")
logger.info("  Total samples: %d", len(X_zeroday))
logger.info("  Detected as anomaly: %d (%.1f%%)", tier1_anomaly.sum(), tier1_anomaly.mean() * 100)

if tier1_anomaly.sum() > 0:
    X_zeroday_t2 = X_zeroday[tier1_anomaly]
    X_zeroday_t2_scaled = scaler_t2.transform(X_zeroday_t2)

    t2_proba = cat_model.predict_proba(X_zeroday_t2_scaled)
    t2_max_proba = np.max(t2_proba, axis=1)
    t2_predictions = cat_model.predict(X_zeroday_t2_scaled)

    unknown_mask = t2_max_proba < UNKNOWN_THRESHOLD
    logger.info("Tier-2 on Detected Anomalies:")
    logger.info("  Unknown: %d (%.1f%%)", unknown_mask.sum(), unknown_mask.mean() * 100)
    logger.info("  Classified: %d (%.1f%%)", (~unknown_mask).sum(), (~unknown_mask).mean() * 100)

    if (~unknown_mask).sum() > 0:
        classified = t2_predictions[~unknown_mask]
        classified_labels = label_encoder.inverse_transform(classified.astype(int))
        logger.info("  Misclassified as: %s", dict(pd.Series(classified_labels).value_counts()))

logger.info("--- Full Pipeline Summary ---")
logger.info("Zero-day samples: %d", len(X_zeroday))
logger.info("Tier-1 detected: %d anomalies", tier1_anomaly.sum())
if tier1_anomaly.sum() > 0:
    logger.info("Tier-2 classified as Unknown: %d", unknown_mask.sum())
    logger.info("Correctly identified as unknown: %.1f%%", unknown_mask.sum() / tier1_anomaly.sum() * 100)

logger.info("Computing SHAP explanations...")
cat_explainer = shap.TreeExplainer(cat_model)
sample_size = min(200, len(X_test_scaled))
cat_shap_values = cat_explainer.shap_values(X_test_scaled[:sample_size])

plt.figure(figsize=(12, 8))
shap.summary_plot(cat_shap_values, X_test[:sample_size], plot_type="bar", show=False)
plt.title("Tier-2 CatBoost Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "tier2_shap_summary.png"), dpi=150)
plt.close()
logger.info("SHAP plot saved.")

joblib.dump(cat_explainer, os.path.join(MODELS_DIR, "tier2_shap_explainer.joblib"))

logger.info("Saving models and artifacts...")
joblib.dump(cat_model, os.path.join(MODELS_DIR, "tier2_catboost.joblib"))
joblib.dump(scaler_t2, os.path.join(MODELS_DIR, "tier2_preprocessor.joblib"))
joblib.dump(label_encoder, os.path.join(MODELS_DIR, "tier2_label_encoder.joblib"))
joblib.dump(feature_order, os.path.join(MODELS_DIR, "tier2_feature_order.joblib"))

thresholds = {
    "unknown_threshold": UNKNOWN_THRESHOLD,
    "classes": list(label_encoder.classes_),
}
joblib.dump(thresholds, os.path.join(MODELS_DIR, "tier2_thresholds.joblib"))

config["tier2"]["classes"] = list(label_encoder.classes_)
with open(os.path.join(base, "config", "config.json"), "w") as f:
    json.dump(config, f, indent=4)

logger.info("Tier-2 training complete. Models saved to %s", MODELS_DIR)
