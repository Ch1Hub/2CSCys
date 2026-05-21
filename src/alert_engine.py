import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertEngine:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config.get("alert", {}).get("output_dir", "output")
        self.log_dir = config.get("alert", {}).get("log_dir", "logs")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def generate_alert(self, tier1_result: dict, tier2_result: dict,
                       unknown_result: dict, explanation: dict,
                       source: str = "pcap", window_id: int = 0) -> dict:
        status = "benign" if tier1_result.get("status") == "benign" else "malicious"

        if status == "benign":
            attack = "N/A"
            confidence = tier1_result.get("probability", 0.0)
        else:
            attack = unknown_result.get("attack", "Unknown")
            confidence = unknown_result.get("confidence", 0.0)

        alert = {
            "status": status,
            "attack": attack,
            "confidence": round(confidence, 4),
            "source": source,
            "window_id": window_id,
            "explanation": explanation,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tier1_detail": tier1_result,
            "tier2_detail": tier2_result if status == "malicious" else None,
            "unknown_detection": unknown_result if status == "malicious" else None
        }

        return alert

    def save_alert(self, alert: dict):
        ts = alert.get("timestamp", datetime.utcnow().isoformat()).replace(":", "-")
        filename = f"alert_{ts}_{alert.get('window_id', 0)}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(alert, f, indent=2, default=str)

        logger.info("Alert saved to %s", filepath)
        return filepath

    def log_alert(self, alert: dict):
        status = alert.get("status", "unknown")
        attack = alert.get("attack", "N/A")
        confidence = alert.get("confidence", 0.0)
        src = alert.get("source", "unknown")
        wid = alert.get("window_id", 0)

        log_line = (
            f"[ALERT] status={status} attack={attack} "
            f"confidence={confidence:.4f} source={src} window_id={wid}"
        )

        if status == "malicious":
            logger.warning(log_line)
        else:
            logger.info(log_line)

        log_path = os.path.join(self.log_dir, "alerts.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(alert, default=str) + "\n")