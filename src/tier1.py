import logging
import os

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Tier1Classifier:
    def __init__(self, config: dict):
        self.config = config
        tier1_cfg = config.get("tier1", {})
        self.lgbm_threshold = tier1_cfg.get("lgbm_threshold", 0.30)
        self.use_iforest = tier1_cfg.get("use_iforest", True)
        self.lgbm_model = None
        self.iforest_model = None
        self.preprocessor = None
        self.feature_order = None
        self.lgbm_path = tier1_cfg.get("model_path", "models/tier1_lgbm.joblib")
        self.iforest_path = tier1_cfg.get("iforest_path", "models/tier1_iforest.joblib")
        self.preprocessor_path = tier1_cfg.get("preprocessor_path", "models/tier1_preprocessor.joblib")
        self.feature_order_path = tier1_cfg.get("feature_order_path", "models/tier1_feature_order.joblib")

    def load(self, model_dir: str = None):
        base = model_dir or "models"
        self.lgbm_model = joblib.load(os.path.join(base, os.path.basename(self.lgbm_path)))
        self.preprocessor = joblib.load(os.path.join(base, os.path.basename(self.preprocessor_path)))
        self.feature_order = joblib.load(os.path.join(base, os.path.basename(self.feature_order_path)))
        if self.use_iforest:
            try:
                self.iforest_model = joblib.load(os.path.join(base, os.path.basename(self.iforest_path)))
            except FileNotFoundError:
                logger.warning("IsoForest model not found at %s, using LGBM-only mode", self.iforest_path)
                self.iforest_model = None
        else:
            self.iforest_model = None
            logger.info("IsoForest disabled in config, using LGBM-only mode")
        logger.info("Tier-1 models loaded from %s (IsoForest: %s)", base, "enabled" if self.iforest_model else "disabled")

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.feature_order is not None:
            for col in self.feature_order:
                if col not in features.columns:
                    features[col] = 0
            features = features[self.feature_order]
        return features

    def _predict_core(self, features: pd.DataFrame) -> list:
        if features.empty:
            return [{"status": "benign", "probability": 0.0}]

        features = self._align_features(features)

        lgbm_proba = self.lgbm_model.predict_proba(features)
        anomaly_proba = lgbm_proba[:, 1] if lgbm_proba.shape[1] > 1 else lgbm_proba[:, 0]

        use_iforest = self.iforest_model is not None
        if use_iforest:
            X_scaled = self.preprocessor.transform(features)
            iforest_pred = self.iforest_model.predict(X_scaled)
            iforest_anomaly = (iforest_pred == -1)
            lgbm_threshold = self.lgbm_threshold
        else:
            lgbm_threshold = self.config.get("tier1", {}).get("lgbm_threshold_nof", self.lgbm_threshold)

        results = []
        for i in range(len(features)):
            lgbm_score = float(anomaly_proba[i])

            is_anomaly = lgbm_score > lgbm_threshold
            if use_iforest and not is_anomaly:
                is_anomaly = bool(iforest_anomaly[i])

            if is_anomaly:
                status = "anomaly"
                if use_iforest:
                    prob = max(lgbm_score, 0.5 if iforest_anomaly[i] else lgbm_score)
                else:
                    prob = lgbm_score
            else:
                status = "benign"
                prob = 1.0 - lgbm_score

            results.append({
                "status": status,
                "probability": round(prob, 4)
            })

        return results

    def predict(self, features: pd.DataFrame) -> dict:
        results = self._predict_core(features)
        return results[0] if len(results) == 1 else results

    def predict_batch(self, features: pd.DataFrame) -> list:
        return self._predict_core(features)