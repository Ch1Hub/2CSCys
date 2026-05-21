import json
import logging
import os
import time
from typing import Optional

import pandas as pd

from src.zeek_runner import ZeekRunner
from src.log_parser import parse_all_logs
from src.feature_extractor import FeatureExtractor
from src.window_manager import WindowManager
from src.tier1 import Tier1Classifier
from src.tier2 import Tier2Classifier
from src.unknown_detector import UnknownDetector
from src.alert_engine import AlertEngine
from src.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.json") -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


class OfflinePipeline:
    def __init__(self, config: dict):
        self.config = config
        self.zeek_runner = ZeekRunner(config)
        self.feature_extractor = FeatureExtractor(config)
        self.tier1 = Tier1Classifier(config)
        self.tier2 = Tier2Classifier(config)
        self.unknown_detector = UnknownDetector(config)
        self.alert_engine = AlertEngine(config)
        self.shap_explainer = SHAPExplainer(config)

    def load_models(self, model_dir: str = "models"):
        self.tier1.load(model_dir)
        self.tier2.load(model_dir)
        self._load_shap(model_dir)
        logger.info("All models loaded for offline pipeline")

    def _load_shap(self, model_dir: str = "models"):
        import joblib
        try:
            t1_model = self.tier1.lgbm_model
            feat_order = self.tier1.feature_order
            self.shap_explainer.load_tier1(t1_model, feat_order)
            logger.info("Tier-1 SHAP explainer loaded")
        except Exception as e:
            logger.warning("Tier-1 SHAP explainer not loaded: %s", e)
        try:
            t2_model = self.tier2.model
            feat_order = self.tier2.feature_order
            self.shap_explainer.load_tier2(t2_model, feat_order)
            logger.info("Tier-2 SHAP explainer loaded")
        except Exception as e:
            logger.warning("Tier-2 SHAP explainer not loaded: %s", e)

    def process_pcap(self, pcap_path: str, output_dir: Optional[str] = None) -> list:
        logger.info("Starting offline pipeline for %s", pcap_path)

        log_dir = self.zeek_runner.run_offline(pcap_path, output_dir)
        log_paths = self.zeek_runner.get_log_paths(log_dir)

        logs = parse_all_logs(log_paths)
        if "conn" not in logs:
            logger.error("No conn.log found - cannot proceed")
            return []

        features = self.feature_extractor.build_feature_vector(logs)
        if features.empty:
            logger.error("Feature extraction produced empty result")
            return []

        tier1_results = self.tier1.predict_batch(features)
        results = []

        anomaly_mask = [r["status"] == "anomaly" for r in tier1_results]

        tier2_results = [None] * len(tier1_results)
        unknown_results = [None] * len(tier1_results)
        explanations = [None] * len(tier1_results)

        if any(anomaly_mask):
            anomaly_features = features[anomaly_mask].reset_index(drop=True)
            tier2_anomaly = self.tier2.predict_batch(anomaly_features)

            idx = 0
            for i, is_anomaly in enumerate(anomaly_mask):
                if is_anomaly:
                    tier2_results[i] = tier2_anomaly[idx]
                    unknown_results[i] = self.unknown_detector.detect(tier2_anomaly[idx])
                    idx += 1

        for i in range(len(tier1_results)):
            if tier1_results[i]["status"] == "benign":
                explanations[i] = {"top_features": [], "note": "Benign - no explanation needed"}
            elif tier2_results[i] is not None:
                anomaly_idx = sum(1 for j in range(i) if anomaly_mask[j])
                single_row = features[anomaly_mask].iloc[[anomaly_idx]]
                try:
                    explanations[i] = self.shap_explainer.explain_tier2(single_row)
                except Exception as e:
                    logger.warning("SHAP explanation failed: %s", e)
                    explanations[i] = {"top_features": [], "note": "SHAP failed"}

        for i in range(len(tier1_results)):
            t1 = tier1_results[i]
            t2 = tier2_results[i] or {"attack": "N/A", "confidence": 0.0}
            unk = unknown_results[i] or {"attack": "N/A", "confidence": 0.0, "is_unknown": False}
            exp = explanations[i] or {"top_features": []}

            alert = self.alert_engine.generate_alert(
                tier1_result=t1,
                tier2_result=t2,
                unknown_result=unk,
                explanation=exp,
                source="pcap",
                window_id=i
            )
            results.append(alert)

        logger.info("Offline pipeline complete: %d records processed", len(results))
        return results


class LivePipeline:
    def __init__(self, config: dict):
        self.config = config
        self.zeek_runner = ZeekRunner(config)
        self.feature_extractor = FeatureExtractor(config)
        self.tier1 = Tier1Classifier(config)
        self.tier2 = Tier2Classifier(config)
        self.unknown_detector = UnknownDetector(config)
        self.alert_engine = AlertEngine(config)
        self.shap_explainer = SHAPExplainer(config)
        self.window_manager = WindowManager(config)
        self.running = False
        self.window_id_counter = 0

    def load_models(self, model_dir: str = "models"):
        self.tier1.load(model_dir)
        self.tier2.load(model_dir)
        self._load_shap(model_dir)
        logger.info("All models loaded for live pipeline")

    def process_window(self, conn_row: dict) -> dict:
        self.window_id_counter += 1
        wid = self.window_id_counter

        feature_row = self.window_manager.build_feature_row(conn_row)
        feature_df = pd.DataFrame([feature_row])

        tier1_result = self.tier1.predict(feature_df)

        if tier1_result["status"] == "benign":
            alert = self.alert_engine.generate_alert(
                tier1_result=tier1_result,
                tier2_result={"attack": "N/A", "confidence": 0.0},
                unknown_result={"attack": "N/A", "confidence": 0.0, "is_unknown": False},
                explanation={"top_features": []},
                source="live",
                window_id=wid
            )
            return alert

        tier2_result = self.tier2.predict(feature_df)
        unknown_result = self.unknown_detector.detect(tier2_result)

        try:
            explanation = self.shap_explainer.explain_tier2(feature_df)
        except Exception as e:
            logger.warning("SHAP explanation failed: %s", e)
            explanation = {"top_features": []}

        alert = self.alert_engine.generate_alert(
            tier1_result=tier1_result,
            tier2_result=tier2_result,
            unknown_result=unknown_result,
            explanation=explanation,
            source="live",
            window_id=wid
        )
        return alert

    def start(self, interface: str, duration: Optional[int] = None):
        logger.info("Starting live pipeline on interface %s", interface)
        self.running = True

        proc = self.zeek_runner.run_live(interface)

        try:
            window_cfg = self.config.get("window", {})
            agg_window = window_cfg.get("aggregation_window_seconds", 30)
            check_interval = window_cfg.get("check_interval_seconds", 5)

            while self.running:
                time.sleep(check_interval)
                self._process_accumulated_logs()

                if duration and self.window_id_counter * check_interval >= duration:
                    logger.info("Duration reached, stopping live capture")
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.zeek_runner.stop_live(proc)
            self.running = False

    def stop(self):
        self.running = False

    def _process_accumulated_logs(self):
        log_dir = self.config.get("zeek", {}).get("output_dir", "logs")
        log_paths = self.zeek_runner.get_log_paths(log_dir)

        if "conn" not in log_paths:
            return

        logs = parse_all_logs(log_paths)
        if not logs or logs.get("conn") is None or logs["conn"].empty:
            return

        features = self.feature_extractor.build_feature_vector(logs)
        if features.empty:
            return

        new_rows = features[features["ts"] > getattr(self, "_last_ts", 0)]
        if new_rows.empty:
            return

        for _, row in new_rows.iterrows():
            conn_row = row.to_dict()
            alert = self.process_window(conn_row)
            self.alert_engine.log_alert(alert)

        self._last_ts = new_rows["ts"].max()