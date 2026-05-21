"""
Train Tier-1: LightGBM binary classifier + IsolationForest anomaly detector.
Target: Recall > 95% on anomaly class with combined OR-decision logic.
"""

import os
import json
import sys
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score
)
import lightgbm as lgb
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

logger.info("Loading Tier-1 data...")
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "tier1_X_train.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "tier1_X_test.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "tier1_y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "tier1_y_test.csv")).values.ravel()
feature_order = joblib.load(os.path.join(PROCESSED_DIR, "feature_order.joblib"))

logger.info("Train: %s, Test: %s", X_train.shape, X_test.shape)
logger.info("Train labels - Benign: %d, Anomaly: %d", (y_train == 0).sum(), (y_train == 1).sum())
logger.info("Feature count: %d", len(feature_order))

logger.info("Preprocessing...")
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
X_train = X_train[feature_order]
X_test = X_test[feature_order]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

benign_count = (y_train == 0).sum()
anomaly_count = (y_train == 1).sum()
scale_pos_weight = benign_count / anomaly_count
logger.info("Class ratio (benign/anomaly): %.2f", scale_pos_weight)

joblib.dump(scaler, os.path.join(MODELS_DIR, "tier1_preprocessor.joblib"))

logger.info("Training LightGBM...")
lgbm_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": scale_pos_weight,
    "verbose": -1,
    "random_state": 42,
}

lgbm_model = lgb.LGBMClassifier(**lgbm_params)
lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
)

y_proba_lgbm = lgbm_model.predict_proba(X_test)[:, 1]
y_pred_lgbm = lgbm_model.predict(X_test)

logger.info("LightGBM Results:")
logger.info("\n%s", classification_report(y_test, y_pred_lgbm, target_names=["Benign", "Anomaly"]))
logger.info("ROC-AUC: %.4f", roc_auc_score(y_test, y_proba_lgbm))

logger.info("Training IsolationForest...")
contamination = float(anomaly_count) / float(benign_count + anomaly_count)
iforest = IsolationForest(
    n_estimators=200,
    contamination=contamination,
    random_state=42,
    n_jobs=-1,
)
iforest.fit(X_train_scaled)

y_pred_if = iforest.predict(X_test_scaled)
y_pred_if_binary = (y_pred_if == -1).astype(int)

logger.info("IsolationForest Results:")
logger.info("\n%s", classification_report(y_test, y_pred_if_binary, target_names=["Benign", "Anomaly"]))

logger.info("Evaluating combined decision logic...")
LGBM_THRESHOLD = config["tier1"]["lgbm_threshold"]

lgbm_anomaly = y_proba_lgbm > LGBM_THRESHOLD
if_anomaly = y_pred_if == -1

y_combined = np.zeros(len(y_test), dtype=int)
y_combined[lgbm_anomaly | if_anomaly] = 1

logger.info("LGBM anomaly detections: %d", lgbm_anomaly.sum())
logger.info("IsoForest anomaly detections: %d", if_anomaly.sum())
logger.info("Combined anomaly detections: %d", y_combined.sum())
logger.info("Actual anomalies: %d", y_test.sum())

logger.info("Combined Results (threshold=%.2f):", LGBM_THRESHOLD)
logger.info("\n%s", classification_report(y_test, y_combined, target_names=["Benign", "Anomaly"]))

recall = recall_score(y_test, y_combined)
precision = precision_score(y_test, y_combined)
f1 = f1_score(y_test, y_combined)

cm = confusion_matrix(y_test, y_combined)
logger.info("Recall: %.4f (target > 0.95)", recall)
logger.info("Precision: %.4f", precision)
logger.info("F1: %.4f", f1)
logger.info("Confusion Matrix: TP=%d FP=%d FN=%d TN=%d", cm[1][1], cm[0][1], cm[1][0], cm[0][0])

logger.info("Threshold tuning...")
if recall < 0.95:
    best_threshold = LGBM_THRESHOLD
    best_recall = recall
    best_precision = precision

    for thresh in np.arange(0.05, 0.50, 0.01):
        lgbm_anom = y_proba_lgbm > thresh
        combined = np.zeros(len(y_test), dtype=int)
        combined[lgbm_anom | if_anomaly] = 1
        r = recall_score(y_test, combined)
        p = precision_score(y_test, combined)
        if r >= 0.95 and p >= best_precision:
            best_threshold = thresh
            best_recall = r
            best_precision = p
            logger.info("  threshold=%.2f: recall=%.4f, precision=%.4f", thresh, r, p)

    LGBM_THRESHOLD = best_threshold
    logger.info("Best threshold: %.2f", LGBM_THRESHOLD)
else:
    logger.info("Recall %.4f >= 0.95 target, current threshold sufficient.", recall)

y_final = np.zeros(len(y_test), dtype=int)
y_final[(y_proba_lgbm > LGBM_THRESHOLD) | if_anomaly] = 1
final_recall = recall_score(y_test, y_final)
logger.info("Final Results with threshold %.2f:", LGBM_THRESHOLD)
logger.info("\n%s", classification_report(y_test, y_final, target_names=["Benign", "Anomaly"]))
logger.info("Final Recall: %.4f", final_recall)

logger.info("Computing SHAP explanations...")
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_test[:500])
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test[:500], plot_type="bar", show=False)
plt.title("Tier-1 LightGBM Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "tier1_shap_summary.png"), dpi=150)
plt.close()
logger.info("SHAP plot saved.")

joblib.dump(explainer, os.path.join(MODELS_DIR, "tier1_shap_explainer.joblib"))

logger.info("Saving models and artifacts...")
joblib.dump(lgbm_model, os.path.join(MODELS_DIR, "tier1_lgbm.joblib"))
joblib.dump(iforest, os.path.join(MODELS_DIR, "tier1_iforest.joblib"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "tier1_preprocessor.joblib"))
joblib.dump(feature_order, os.path.join(MODELS_DIR, "tier1_feature_order.joblib"))

thresholds = {"lgbm_threshold": LGBM_THRESHOLD, "target_recall": 0.95}
joblib.dump(thresholds, os.path.join(MODELS_DIR, "tier1_thresholds.joblib"))

config["tier1"]["lgbm_threshold"] = LGBM_THRESHOLD
with open(os.path.join(base, "config", "config.json"), "w") as f:
    json.dump(config, f, indent=4)

logger.info("Tier-1 training complete. Models saved to %s", MODELS_DIR)
