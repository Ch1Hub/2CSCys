import logging
import time
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WindowManager:
    def __init__(self, config: dict):
        self.config = config
        window_cfg = config.get("window", {})
        self.short_window = window_cfg.get("short_window_seconds", 5)
        self.agg_window = window_cfg.get("aggregation_window_seconds", 30)
        self.buffers = {
            "short": [],
            "aggregation": []
        }
        self.window_id = 0

    def add_connection(self, conn_row: dict):
        self.buffers["short"].append(conn_row)
        self.buffers["aggregation"].append(conn_row)
        self._prune_buffers()

    def _prune_buffers(self):
        now = time.time()
        self.buffers["short"] = [
            r for r in self.buffers["short"]
            if now - r.get("ts", 0) <= self.short_window
        ]
        self.buffers["aggregation"] = [
            r for r in self.buffers["aggregation"]
            if now - r.get("ts", 0) <= self.agg_window
        ]

    def get_short_window_features(self) -> dict:
        buffer = self.buffers["short"]
        return self._compute_window_stats(buffer)

    def get_agg_window_features(self) -> dict:
        buffer = self.buffers["aggregation"]
        return self._compute_window_stats(buffer)

    def _compute_window_stats(self, buffer: list) -> dict:
        if not buffer:
            return {
                "connections_count": 0,
                "unique_dst_ips": 0,
                "unique_dst_ports": 0,
                "failed_connections": 0,
                "avg_duration": 0.0,
                "total_orig_bytes": 0.0,
                "total_resp_bytes": 0.0,
                "total_orig_pkts": 0.0,
                "total_resp_pkts": 0.0,
                "proto_distribution": {},
                "conn_state_distribution": {},
                "service_distribution": {}
            }

        failed_states = {"REJ", "RSTO", "RSTR", "RSTOS0", "S0", "SH"}

        dst_ips = set()
        dst_ports = set()
        failed = 0
        total_duration = 0.0
        total_orig_bytes = 0.0
        total_resp_bytes = 0.0
        total_orig_pkts = 0.0
        total_resp_pkts = 0.0
        proto_dist = defaultdict(int)
        conn_state_dist = defaultdict(int)
        service_dist = defaultdict(int)

        for row in buffer:
            dst_ips.add(row.get("dst_ip", ""))
            dst_port = row.get("dst_port", 0)
            if dst_port:
                dst_ports.add(dst_port)
            if row.get("conn_state", "") in failed_states:
                failed += 1
            total_duration += float(row.get("duration", 0))
            total_orig_bytes += float(row.get("orig_bytes", 0))
            total_resp_bytes += float(row.get("resp_bytes", 0))
            total_orig_pkts += float(row.get("orig_pkts", 0))
            total_resp_pkts += float(row.get("resp_pkts", 0))
            proto_dist[row.get("proto", "")] += 1
            conn_state_dist[row.get("conn_state", "")] += 1
            service_dist[row.get("service", "")] += 1

        n = len(buffer)
        return {
            "connections_count": n,
            "unique_dst_ips": len(dst_ips),
            "unique_dst_ports": len(dst_ports),
            "failed_connections": failed,
            "avg_duration": total_duration / n,
            "total_orig_bytes": total_orig_bytes,
            "total_resp_bytes": total_resp_bytes,
            "total_orig_pkts": total_orig_pkts,
            "total_resp_pkts": total_resp_pkts,
            "proto_distribution": dict(proto_dist),
            "conn_state_distribution": dict(conn_state_dist),
            "service_distribution": dict(service_dist)
        }

    def build_feature_row(self, conn_row: dict) -> dict:
        self.add_connection(conn_row)

        short_stats = self.get_short_window_features()
        agg_stats = self.get_agg_window_features()

        feature_row = {k: v for k, v in conn_row.items()}

        for k in ["uid", "ts"]:
            if k not in feature_row:
                feature_row[k] = conn_row.get(k, "" if k == "uid" else time.time())

        numeric_defaults = {
            "duration": 0.0, "orig_bytes": 0.0, "resp_bytes": 0.0,
            "orig_pkts": 0.0, "resp_pkts": 0.0, "dst_port": 0,
            "dns_entropy": 0.0, "nxdomain_ratio": 0.0,
            "uri_length": 0.0, "response_code": 0.0, "user_agent_entropy": 0.0,
            "ja3_hash": 0, "cipher_count": 0.0, "self_signed": 0,
        }
        for k, default in numeric_defaults.items():
            if k not in feature_row or feature_row[k] is None:
                feature_row[k] = default
            try:
                feature_row[k] = float(feature_row[k])
            except (ValueError, TypeError):
                feature_row[k] = default

        str_defaults = {
            "service": "-", "conn_state": "-", "proto": "-",
            "method": "GET", "tls_version": "unknown",
        }
        for k, default in str_defaults.items():
            if k not in feature_row or feature_row[k] is None:
                feature_row[k] = default

        duration = feature_row.get("duration", 0.0) or 0.001
        orig_bytes = feature_row.get("orig_bytes", 0.0)
        resp_bytes = feature_row.get("resp_bytes", 0.0)
        orig_pkts = feature_row.get("orig_pkts", 0.0)
        resp_pkts = feature_row.get("resp_pkts", 0.0)

        feature_row["flow_rate"] = (orig_bytes + resp_bytes) / duration
        feature_row["bytes_ratio"] = orig_bytes / (resp_bytes + 1)
        feature_row["packets_ratio"] = orig_pkts / (resp_pkts + 1)

        feature_row["connections_count_5s"] = short_stats["connections_count"]
        feature_row["connections_count_30s"] = agg_stats["connections_count"]
        feature_row["unique_dst_ips"] = agg_stats["unique_dst_ips"]
        feature_row["unique_dst_ports"] = agg_stats["unique_dst_ports"]
        feature_row["failed_connections"] = agg_stats["failed_connections"]

        import math as _m
        for ratio_key in ["flow_rate", "bytes_ratio", "packets_ratio"]:
            val = feature_row.get(ratio_key, 0.0)
            if val == float("inf") or val == float("-inf") or (_m.isnan(val) if isinstance(val, float) else False):
                feature_row[ratio_key] = 0.0

        self.window_id += 1
        return feature_row

    def reset(self):
        self.buffers = {"short": [], "aggregation": []}
        self.window_id = 0