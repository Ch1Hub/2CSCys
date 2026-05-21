import logging
import os

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Tier2Classifier:
    def __init__(self, config: dict):
        self.config = config
        tier2_cfg = config.get("tier2", {})
        self.unknown_threshold = tier2_cfg.get("unknown_threshold", 0.65)
        self.classes = tier2_cfg.get("classes", ["DoS", "BruteForce", "PortScan"])
        self.model = None
        self.preprocessor = None
        self.feature_order = None
        self.label_encoder = None
        self.model_path = tier2_cfg.get("model_path", "models/tier2_catboost.joblib")
        self.preprocessor_path = tier2_cfg.get("preprocessor_path", "models/tier2_preprocessor.joblib")
        self.feature_order_path = tier2_cfg.get("feature_order_path", "models/tier2_feature_order.joblib")
        self.label_encoder_path = tier2_cfg.get("label_encoder_path", "models/tier2_label_encoder.joblib")

    def load(self, model_dir: str = None):
        base = model_dir or "models"
        self.model = joblib.load(os.path.join(base, os.path.basename(self.model_path)))
        self.preprocessor = joblib.load(os.path.join(base, os.path.basename(self.preprocessor_path)))
        self.feature_order = joblib.load(os.path.join(base, os.path.basename(self.feature_order_path)))
        try:
            self.label_encoder = joblib.load(os.path.join(base, os.path.basename(self.label_encoder_path)))
            self.classes = list(self.label_encoder.classes_)
        except FileNotFoundError:
            logger.warning("Label encoder not found, using classes from config")
        logger.info("Tier-2 model loaded from %s", base)

    def _pred_to_label(self, pred_val):
        if self.label_encoder is not None:
            try:
                return self.label_encoder.inverse_transform([int(pred_val)])[0]
            except Exception:
                pass
        if str(pred_val).isdigit():
            idx = int(pred_val)
            if idx < len(self.classes):
                return self.classes[idx]
        return str(pred_val)

    def _preprocess(self, features: pd.DataFrame) -> np.ndarray:
        if self.feature_order is not None:
            for col in self.feature_order:
                if col not in features.columns:
                    features[col] = 0
            features = features[self.feature_order]

        if self.preprocessor is not None:
            X = self.preprocessor.transform(features)
        else:
            X = features.values

        return X

    def predict(self, features: pd.DataFrame) -> dict:
        if features.empty:
            return {"attack": "Unknown", "confidence": 0.0}

        X = self._preprocess(features)

        proba = self.model.predict_proba(X)
        predictions = self.model.predict(X).ravel()

        results = []
        for i in range(len(X)):
            max_prob = float(np.max(proba[i]))
            pred_class = self._pred_to_label(predictions[i])

            if max_prob < self.unknown_threshold:
                attack = "Unknown"
            else:
                attack = pred_class

            results.append({
                "attack": attack,
                "confidence": round(max_prob, 4),
                "all_probabilities": {
                    cls: round(float(proba[i][j]), 4)
                    for j, cls in enumerate(self.classes)
                }
            })

        return results if len(results) > 1 else results[0]

    def predict_batch(self, features: pd.DataFrame) -> list:
        if features.empty:
            return [{"attack": "Unknown", "confidence": 0.0}]

        X = self._preprocess(features)

        proba = self.model.predict_proba(X)
        predictions = self.model.predict(X).ravel()

        results = []
        for i in range(len(X)):
            max_prob = float(np.max(proba[i]))
            pred_class = self._pred_to_label(predictions[i])

            if max_prob < self.unknown_threshold:
                attack = "Unknown"
            else:
                attack = pred_class

            results.append({
                "attack": attack,
                "confidence": round(max_prob, 4),
                "all_probabilities": {
                    cls: round(float(proba[i][j]), 4)
                    for j, cls in enumerate(self.classes)
                }
            })

        return results