import logging

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class SHAPExplainer:
    def __init__(self, config: dict):
        self.config = config
        self.tier1_explainer = None
        self.tier2_explainer = None
        self.tier1_feature_order = None
        self.tier2_feature_order = None

    def load_tier1(self, model, feature_order, background_data=None):
        self.tier1_model = model
        self.tier1_feature_order = feature_order
        if background_data is not None:
            self.tier1_explainer = shap.TreeExplainer(model, background_data)
        else:
            self.tier1_explainer = shap.TreeExplainer(model)
        logger.info("SHAP explainer loaded for Tier-1")

    def load_tier2(self, model, feature_order, background_data=None):
        self.tier2_model = model
        self.tier2_feature_order = feature_order
        if background_data is not None:
            self.tier2_explainer = shap.TreeExplainer(model, background_data)
        else:
            self.tier2_explainer = shap.TreeExplainer(model)
        logger.info("SHAP explainer loaded for Tier-2")

    def _align_features(self, features, feature_order):
        if feature_order is not None:
            for col in feature_order:
                if col not in features.columns:
                    features[col] = 0
            features = features[feature_order]
        return features

    def _normalize_shap(self, sv_raw, n_features):
        if isinstance(sv_raw, list):
            if len(sv_raw) > 1:
                sv = np.array(sv_raw[-1])
            else:
                sv = np.array(sv_raw[0])
        else:
            sv = np.array(sv_raw)

        if sv.ndim == 1:
            sv = sv.reshape(1, -1)
        elif sv.ndim == 3:
            pred_class = sv.shape[1] // 2 if sv.shape[1] > 1 else 0
            sv = sv[:, pred_class, :]

        if sv.ndim == 3:
            sv = sv[:, 0, :]

        if sv.shape[-1] != n_features:
            if sv.ndim == 2 and sv.shape[0] == n_features:
                sv = sv.T

        if sv.ndim == 1:
            sv = sv.reshape(1, -1)

        return sv

    def explain_tier1(self, features: pd.DataFrame, top_n: int = 5) -> dict:
        if self.tier1_explainer is None:
            return {"top_features": [], "note": "SHAP explainer not loaded"}

        features = self._align_features(features, self.tier1_feature_order)
        feature_names = list(features.columns)
        n_features = len(feature_names)

        try:
            sv_raw = self.tier1_explainer.shap_values(features)
            sv = self._normalize_shap(sv_raw, n_features)
        except Exception as e:
            logger.error("Tier-1 SHAP computation failed: %s", e)
            return {"top_features": [], "note": f"SHAP failed: {e}"}

        return self._build_explanations(sv, feature_names, top_n)

    def explain_tier2(self, features: pd.DataFrame, top_n: int = 5) -> dict:
        if self.tier2_explainer is None:
            return {"top_features": [], "note": "SHAP explainer not loaded"}

        features = self._align_features(features, self.tier2_feature_order)
        feature_names = list(features.columns)
        n_features = len(feature_names)

        try:
            sv_raw = self.tier2_explainer.shap_values(features)
            sv = self._normalize_shap(sv_raw, n_features)
        except Exception as e:
            logger.error("Tier-2 SHAP computation failed: %s", e)
            return {"top_features": [], "note": f"SHAP failed: {e}"}

        return self._build_explanations(sv, feature_names, top_n)

    def _build_explanations(self, sv, feature_names, top_n):
        explanations = []

        for i in range(sv.shape[0]):
            row = sv[i]
            abs_row = np.abs(row)
            top_indices = np.argsort(abs_row)[-top_n:][::-1]

            top_features = []
            for idx in top_indices:
                idx_int = int(idx)
                feat_name = feature_names[idx_int] if idx_int < len(feature_names) else f"feature_{idx_int}"
                shap_val = float(row[idx_int])
                top_features.append({
                    "feature": feat_name,
                    "importance": round(float(abs_row[idx_int]), 4),
                    "direction": "positive" if shap_val > 0 else "negative",
                    "shap_value": round(shap_val, 4)
                })

            shap_dict = {}
            for j in range(min(len(row), len(feature_names))):
                try:
                    shap_dict[feature_names[j]] = round(float(row[j]), 4)
                except (IndexError, ValueError):
                    pass

            explanations.append({
                "top_features": top_features,
                "shap_values": shap_dict
            })

        if len(explanations) == 1:
            return explanations[0]
        return explanations