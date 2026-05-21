import logging
import math
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _entropy(values):
    if not values:
        return 0.0
    counter = Counter(values)
    total = len(values)
    ent = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def _safe_divide(a, b, default=0.0):
    return a / b if b != 0 else default


class FeatureExtractor:
    def __init__(self, config: dict):
        self.config = config
        self.feature_config = config.get("features", {})
        self.window_config = config.get("window", {})

    def extract_connection_features(self, conn_df: pd.DataFrame) -> pd.DataFrame:
        if conn_df is None or conn_df.empty:
            logger.warning("Empty conn_df for connection features")
            return pd.DataFrame()

        result = conn_df[["uid", "ts"]].copy()

        dur_col = "duration" if "duration" in conn_df.columns else "dur"
        if dur_col in conn_df.columns:
            result["duration"] = pd.to_numeric(conn_df[dur_col], errors="coerce").fillna(0)
        else:
            result["duration"] = 0.0

        for col in ["orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]:
            if col in conn_df.columns:
                result[col] = pd.to_numeric(conn_df[col], errors="coerce").fillna(0)
            else:
                result[col] = 0.0

        for col in ["service", "conn_state", "proto"]:
            if col in conn_df.columns:
                result[col] = conn_df[col].fillna("-").astype(str)
            else:
                result[col] = "-"

        if "id.orig_h" in conn_df.columns:
            result["src_ip"] = conn_df["id.orig_h"]
        if "id.resp_h" in conn_df.columns:
            result["dst_ip"] = conn_df["id.resp_h"]
        if "id.resp_p" in conn_df.columns:
            result["dst_port"] = pd.to_numeric(conn_df["id.resp_p"], errors="coerce").fillna(0)

        return result

    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        duration = df["duration"].replace(0, 0.001)
        df["flow_rate"] = (df["orig_bytes"] + df["resp_bytes"]) / duration
        df["bytes_ratio"] = df["orig_bytes"] / (df["resp_bytes"] + 1)
        df["packets_ratio"] = df["orig_pkts"] / (df["resp_pkts"] + 1)

        df["flow_rate"] = df["flow_rate"].replace([np.inf, -np.inf], 0).fillna(0)
        df["bytes_ratio"] = df["bytes_ratio"].replace([np.inf, -np.inf], 0).fillna(0)
        df["packets_ratio"] = df["packets_ratio"].replace([np.inf, -np.inf], 0).fillna(0)

        return df

    def extract_dns_features(self, dns_df: pd.DataFrame, conn_uids: pd.Series) -> pd.DataFrame:
        result = pd.DataFrame(index=conn_uids.values)
        result["dns_entropy"] = 0.0
        result["nxdomain_ratio"] = 0.0

        if dns_df is None or dns_df.empty:
            return result

        if "uid" in dns_df.columns:
            dns_per_uid = dns_df.groupby("uid")
        else:
            return result

        entropy_map = {}
        nxdomain_map = {}

        for uid, group in dns_per_uid:
            if "query" in group.columns:
                queries = group["query"].dropna().astype(str).tolist()
                if queries:
                    lengths = [len(q) for q in queries]
                    entropy_map[uid] = _entropy(lengths)
                else:
                    entropy_map[uid] = 0.0
            else:
                entropy_map[uid] = 0.0

            if "rcode" in group.columns:
                rcodes = group["rcode"].dropna().astype(str).tolist()
                nx_count = sum(1 for r in rcodes if r in ("3", "NXDOMAIN"))
                total = len(rcodes) if rcodes else 1
                nxdomain_map[uid] = _safe_divide(nx_count, total)
            else:
                nxdomain_map[uid] = 0.0

        result["dns_entropy"] = result.index.map(lambda u: entropy_map.get(u, 0.0))
        result["nxdomain_ratio"] = result.index.map(lambda u: nxdomain_map.get(u, 0.0))

        return result

    def extract_http_features(self, http_df: pd.DataFrame, conn_uids: pd.Series) -> pd.DataFrame:
        result = pd.DataFrame(index=conn_uids.values)
        result["method"] = "GET"
        result["uri_length"] = 0.0
        result["response_code"] = 0.0
        result["user_agent_entropy"] = 0.0

        if http_df is None or http_df.empty:
            return result

        if "uid" not in http_df.columns:
            return result

        method_map = {}
        uri_map = {}
        respcode_map = {}
        ua_entropy_map = {}

        for uid, group in http_df.groupby("uid"):
            if "method" in group.columns:
                method_map[uid] = group["method"].mode().iloc[0] if len(group) > 0 else "GET"
            if "uri" in group.columns:
                uris = group["uri"].dropna().astype(str)
                uri_map[uid] = uris.str.len().mean() if len(uris) > 0 else 0.0
            if "status_code" in group.columns:
                codes = pd.to_numeric(group["status_code"], errors="coerce").dropna()
                respcode_map[uid] = codes.mode().iloc[0] if len(codes) > 0 else 0.0
            if "user_agent" in group.columns:
                uas = group["user_agent"].dropna().astype(str).tolist()
                ua_entropy_map[uid] = _entropy([len(u) for u in uas]) if uas else 0.0

        result["method"] = result.index.map(lambda u: method_map.get(u, "GET"))
        result["uri_length"] = result.index.map(lambda u: uri_map.get(u, 0.0))
        result["response_code"] = result.index.map(lambda u: respcode_map.get(u, 0.0))
        result["user_agent_entropy"] = result.index.map(lambda u: ua_entropy_map.get(u, 0.0))

        return result

    def extract_tls_features(self, ssl_df: pd.DataFrame, conn_uids: pd.Series) -> pd.DataFrame:
        result = pd.DataFrame(index=conn_uids.values)
        result["ja3_hash"] = 0
        result["tls_version"] = "unknown"
        result["cipher_count"] = 0.0
        result["self_signed"] = 0

        if ssl_df is None or ssl_df.empty:
            return result

        if "uid" not in ssl_df.columns:
            return result

        ja3_map = {}
        version_map = {}
        cipher_map = {}
        self_signed_map = {}

        for uid, group in ssl_df.groupby("uid"):
            if "ja3" in group.columns:
                ja3_val = group["ja3"].mode().iloc[0] if len(group) > 0 else "0"
                ja3_map[uid] = abs(hash(str(ja3_val))) % (10 ** 9)
            if "version" in group.columns:
                version_map[uid] = group["version"].mode().iloc[0] if len(group) > 0 else "unknown"
            if "cipher" in group.columns:
                ciphers = group["cipher"].dropna().astype(str).tolist()
                cipher_map[uid] = float(len(set(ciphers)))
            if "validation_status" in group.columns:
                statuses = group["validation_status"].dropna().astype(str).tolist()
                self_signed_map[uid] = 1 if any("self" in s.lower() or "invalid" in s.lower() for s in statuses) else 0

        result["ja3_hash"] = result.index.map(lambda u: ja3_map.get(u, 0))
        result["tls_version"] = result.index.map(lambda u: version_map.get(u, "unknown"))
        result["cipher_count"] = result.index.map(lambda u: cipher_map.get(u, 0.0))
        result["self_signed"] = result.index.map(lambda u: self_signed_map.get(u, 0))

        return result

    def compute_window_features(self, conn_df: pd.DataFrame) -> pd.DataFrame:
        if conn_df is None or conn_df.empty:
            return conn_df

        short_window = self.window_config.get("short_window_seconds", 5)
        agg_window = self.window_config.get("aggregation_window_seconds", 30)

        df = conn_df.copy()
        if "ts" in df.columns:
            df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
            df = df.sort_values("ts").reset_index(drop=True)

        if "ts" not in df.columns or df["ts"].isna().all():
            df["connections_count_5s"] = 0
            df["connections_count_30s"] = 0
            df["unique_dst_ips"] = 0
            df["unique_dst_ports"] = 0
            df["failed_connections"] = 0
            return df

        times = df["ts"].values
        n = len(df)

        start_5s = np.searchsorted(times, times - short_window, side="left")
        start_30s = np.searchsorted(times, times - agg_window, side="left")

        df["connections_count_5s"] = np.arange(n, dtype=np.int32) - start_5s + 1
        df["connections_count_30s"] = np.arange(n, dtype=np.int32) - start_30s + 1

        has_dst_ip = "dst_ip" in df.columns
        has_dst_port = "dst_port" in df.columns
        has_state = "conn_state" in df.columns

        failed_states = frozenset(["REJ", "RSTO", "RSTR", "RSTOS0", "S0", "SH"])

        dst_ip_vals = list(df["dst_ip"].values) if has_dst_ip else None
        dst_port_vals = list(df["dst_port"].values) if has_dst_port else None
        conn_state_vals = list(df["conn_state"].values.astype(str)) if has_state else None

        unique_ips = np.zeros(n, dtype=np.int32)
        unique_ports = np.zeros(n, dtype=np.int32)
        failed = np.zeros(n, dtype=np.int32)

        from collections import Counter

        ip_counter = Counter() if has_dst_ip else None
        port_counter = Counter() if has_dst_port else None
        failed_counter = Counter() if has_state else None

        window_5s_front = 0
        window_30s_front = 0

        for i in range(n):
            t = times[i]

            while window_30s_front < i and times[window_30s_front] < t - agg_window:
                if ip_counter is not None:
                    val = dst_ip_vals[window_30s_front]
                    ip_counter[val] -= 1
                    if ip_counter[val] <= 0:
                        del ip_counter[val]
                if port_counter is not None:
                    val = dst_port_vals[window_30s_front]
                    port_counter[val] -= 1
                    if port_counter[val] <= 0:
                        del port_counter[val]
                if failed_counter is not None:
                    s = conn_state_vals[window_30s_front]
                    if s in failed_states:
                        failed_counter[s] -= 1
                        if failed_counter[s] <= 0:
                            del failed_counter[s]
                window_30s_front += 1

            while window_5s_front < i and times[window_5s_front] < t - short_window:
                window_5s_front += 1

            if ip_counter is not None:
                ip_counter[dst_ip_vals[i]] += 1
            if port_counter is not None:
                port_counter[dst_port_vals[i]] += 1
            if failed_counter is not None:
                s = conn_state_vals[i]
                if s in failed_states:
                    failed_counter[s] += 1

            if ip_counter is not None:
                unique_ips[i] = len(ip_counter)
            if port_counter is not None:
                unique_ports[i] = len(port_counter)
            if failed_counter is not None:
                failed[i] = sum(failed_counter.values())

        df["unique_dst_ips"] = unique_ips
        df["unique_dst_ports"] = unique_ports
        df["failed_connections"] = failed

        return df

    def build_feature_vector(self, logs: dict) -> pd.DataFrame:
        conn_df = logs.get("conn")
        if conn_df is None or conn_df.empty:
            logger.error("No conn.log data - cannot build feature vector")
            return pd.DataFrame()

        conn_features = self.extract_connection_features(conn_df)
        conn_features = self.compute_derived_features(conn_features)
        conn_features = self.compute_window_features(conn_features)

        uid_series = conn_features["uid"] if "uid" in conn_features.columns else pd.Series()

        dns_features = self.extract_dns_features(logs.get("dns"), uid_series)
        http_features = self.extract_http_features(logs.get("http"), uid_series)
        tls_features = self.extract_tls_features(logs.get("ssl"), uid_series)

        for feat_df in [dns_features, http_features, tls_features]:
            feat_df.index.name = "uid"
            feat_df.reset_index(inplace=True)

        merged = conn_features.copy()
        for feat_df in [dns_features, http_features, tls_features]:
            if "uid" in feat_df.columns and not feat_df.empty:
                merged = merged.merge(feat_df, on="uid", how="left", suffixes=("", "_extra"))

        drop_cols = [c for c in merged.columns if c.endswith("_extra")]
        merged = merged.drop(columns=drop_cols, errors="ignore")

        numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            merged[col] = merged[col].fillna(0)

        logger.info("Built feature vector: %d rows, %d columns", len(merged), len(merged.columns))
        return merged