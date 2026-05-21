"""
Generate ML dataset from CIC-IDS2017 PCAPs through Zeek.

Pipeline:
  1. Process each PCAP through Zeek -> logs
  2. Parse Zeek logs -> structured DataFrames
  3. Extract features using feature_extractor
  4. Label each flow using CIC-IDS2017 CSV ground-truth
  5. Save processed datasets for training

Usage:
  python generate_dataset.py --pcap-dir data/pcaps --csv-dir data/csv --output-dir data/processed
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.zeek_runner import ZeekRunner
from src.log_parser import parse_all_logs
from src.feature_extractor import FeatureExtractor
from src.pipeline import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


CIC_CSV_COLUMNS = {
    "source_ip": ["Source IP", " Source IP", "src_ip"],
    "destination_ip": ["Destination IP", " Destination IP", "dst_ip"],
    "source_port": ["Source Port", " Source Port", "src_port"],
    "destination_port": ["Destination Port", " Destination Port", "dst_port"],
    "protocol": ["Protocol", " protocol"],
    "timestamp": ["Timestamp", " timestamp", "Flow Start"],
    "label": ["Label", " label"],
}


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_cic_labels(csv_dir):
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    if not csv_files:
        logger.warning("No CSV files found in %s", csv_dir)
        return pd.DataFrame()

    frames = []
    for f in csv_files:
        path = os.path.join(csv_dir, f)
        logger.info("Loading labels from %s", f)
        df = pd.read_csv(path, low_memory=False, encoding_errors="replace")
        df.columns = df.columns.str.strip()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def map_labels(label_series):
    training_classes = ["Benign", "DoS", "BruteForce", "PortScan"]
    excluded_classes = ["DDoS", "Botnet", "WebAttack"]

    mapping = {}
    for lbl in label_series.unique():
        low = lbl.lower().strip()
        if low in ("benign", "benvolent"):
            mapping[lbl] = "Benign"
        elif "ddos" in low:
            mapping[lbl] = "DDoS"
        elif "dos" in low:
            mapping[lbl] = "DoS"
        elif "patator" in low or "brute" in low:
            mapping[lbl] = "BruteForce"
        elif "portscan" in low or "port scan" in low:
            mapping[lbl] = "PortScan"
        elif "bot" in low:
            mapping[lbl] = "Botnet"
        elif "web" in low:
            mapping[lbl] = "WebAttack"
        else:
            mapping[lbl] = "Other"

    return label_series.map(mapping)


def label_zeek_flows(zeek_features, cic_df):
    src_ip_col = find_column(cic_df, CIC_CSV_COLUMNS["source_ip"])
    dst_ip_col = find_column(cic_df, CIC_CSV_COLUMNS["destination_ip"])
    src_port_col = find_column(cic_df, CIC_CSV_COLUMNS["source_port"])
    dst_port_col = find_column(cic_df, CIC_CSV_COLUMNS["destination_port"])
    proto_col = find_column(cic_df, CIC_CSV_COLUMNS["protocol"])
    label_col = find_column(cic_df, CIC_CSV_COLUMNS["label"])

    # If IP columns are missing, use feature-based fuzzy matching
    if not all([src_ip_col, dst_ip_col, label_col]):
        logger.warning("Missing Source/Destination IP columns. Falling back to feature-based matching.")
        logger.warning("Available columns: %s", list(cic_df.columns))
        has_dst_port = "Destination Port" in cic_df.columns and "dst_port" in zeek_features.columns
        has_flows = ("Total Length of Fwd Packets" in cic_df.columns and
                     "Total Length of Bwd Packets" in cic_df.columns and
                     "Total Fwd Packets" in cic_df.columns and
                     "Total Backward Packets" in cic_df.columns and
                     "Flow Duration" in cic_df.columns)
        if has_dst_port and has_flows:
            logger.info("Using feature-based nearest-neighbor matching (dst_port + duration + bytes + pkts)")
            return label_zeek_flows_nn(zeek_features, cic_df)
        else:
            logger.error("Cannot match: need either (Source IP, Destination IP) or (Destination Port + Flow features)")
            return zeek_features

    # Original IP-based matching

    cic_labels = cic_df[[src_ip_col, dst_ip_col, label_col]].copy()
    if src_port_col:
        cic_labels["src_port"] = pd.to_numeric(cic_df[src_port_col], errors="coerce")
    if dst_port_col:
        cic_labels["dst_port"] = pd.to_numeric(cic_df[dst_port_col], errors="coerce")
    cic_labels = cic_labels.dropna(subset=[src_ip_col, dst_ip_col])
    cic_labels["mapped_label"] = map_labels(cic_df[label_col])

    label_map = {}
    for _, row in cic_labels.iterrows():
        key = (str(row[src_ip_col]), str(row[dst_ip_col]))
        label_map[key] = row["mapped_label"]

    def get_label(row):
        src = str(row.get("src_ip", row.get("id.orig_h", "")))
        dst = str(row.get("dst_ip", row.get("id.resp_h", "")))
        label = label_map.get((src, dst), "Benign")
        return label

    zeek_features["mapped_label"] = zeek_features.apply(get_label, axis=1)

    stats = zeek_features["mapped_label"].value_counts()
    logger.info("Label distribution after matching:\n%s", stats.to_string())

    return zeek_features


def label_zeek_flows_nn(zeek_features, cic_df):
    """Label Zeek flows using nearest-neighbor matching on (dst_port, duration, bytes, packets)."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    zeek_features = zeek_features.copy()
    zeek_features["total_bytes"] = zeek_features["orig_bytes"] + zeek_features["resp_bytes"]
    zeek_features["total_pkts"] = zeek_features["orig_pkts"] + zeek_features["resp_pkts"]

    cic_df = cic_df.copy()
    cic_df["total_bytes"] = cic_df["Total Length of Fwd Packets"] + cic_df["Total Length of Bwd Packets"]
    cic_df["total_pkts"] = cic_df["Total Fwd Packets"] + cic_df["Total Backward Packets"]
    cic_df["duration_sec"] = cic_df["Flow Duration"] / 1_000_000.0
    cic_df["mapped_label"] = map_labels(cic_df["Label"])

    z_features = ["dst_port", "duration", "total_bytes", "total_pkts"]
    c_features = ["Destination Port", "duration_sec", "total_bytes", "total_pkts"]

    z_X = zeek_features[z_features].values.astype(np.float64)
    c_X = cic_df[c_features].values.astype(np.float64)
    c_labels = cic_df["mapped_label"].values

    z_X = np.nan_to_num(z_X, nan=0.0, posinf=0.0, neginf=0.0)
    c_X = np.nan_to_num(c_X, nan=0.0, posinf=0.0, neginf=0.0)

    if len(c_X) == 0:
        zeek_features["mapped_label"] = "Benign"
        return zeek_features

    scaler = StandardScaler()
    all_X = np.vstack([z_X, c_X])
    scaler.fit(all_X)
    z_X_norm = scaler.transform(z_X)
    c_X_norm = scaler.transform(c_X)

    logger.info("  NN matching: %d Zeek flows vs %d CIC flows", len(z_X), len(c_X))
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=-1)
    nn.fit(c_X_norm)
    _, indices = nn.kneighbors(z_X_norm)

    zeek_features["mapped_label"] = c_labels[indices.flatten()]
    stats = zeek_features["mapped_label"].value_counts()
    logger.info("Label distribution after NN matching:\n%s", stats.to_string())
    return zeek_features


def process_pcap(pcap_path, config, output_dir):
    zeek_runner = ZeekRunner(config)
    feature_extractor = FeatureExtractor(config)

    pcap_name = os.path.splitext(os.path.basename(pcap_path))[0]
    log_dir = os.path.join(output_dir, "zeek_logs", pcap_name)
    os.makedirs(log_dir, exist_ok=True)

    logger.info("Processing PCAP: %s", pcap_path)
    zeek_runner.run_offline(pcap_path, log_dir)

    log_paths = zeek_runner.get_log_paths(log_dir)
    if "conn" not in log_paths:
        logger.error("No conn.log produced for %s", pcap_path)
        return None

    logs = parse_all_logs(log_paths)
    features = feature_extractor.build_feature_vector(logs)

    if features.empty:
        logger.warning("Empty features for %s", pcap_path)
        return None

    features["pcap_source"] = pcap_name
    logger.info("Extracted %d flows from %s", len(features), pcap_name)
    return features


def main():
    parser = argparse.ArgumentParser(description="Generate ML dataset from CIC-IDS2017 PCAPs via Zeek")
    parser.add_argument("--config", default="config/config.json", help="Config file")
    parser.add_argument("--pcap-dir", default="data/pcaps", help="Directory containing PCAP files")
    parser.add_argument("--csv-dir", default="data/csv", help="Directory containing CIC-IDS2017 CSV label files")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    pcap_files = [f for f in os.listdir(args.pcap_dir) if f.endswith((".pcap", ".pcapng"))]
    if not pcap_files:
        logger.warning("No PCAP files found in %s. Place CIC-IDS2017 PCAPs there.", args.pcap_dir)
        logger.warning("Download from: https://www.unb.ca/cic/datasets/ids-2017.html")
        return

    logger.info("Found %d PCAP files", len(pcap_files))

    all_features = []
    for pcap_file in sorted(pcap_files):
        pcap_path = os.path.join(args.pcap_dir, pcap_file)
        features = process_pcap(pcap_path, config, args.pcap_dir)
        if features is not None:
            all_features.append(features)

    if not all_features:
        logger.error("No features extracted from any PCAP")
        return

    combined = pd.concat(all_features, ignore_index=True)
    logger.info("Total flows: %d", len(combined))

    csv_dir = args.csv_dir
    if os.path.isdir(csv_dir) and any(f.endswith(".csv") for f in os.listdir(csv_dir)):
        logger.info("Loading CIC-IDS2017 labels from %s", csv_dir)
        cic_df = load_cic_labels(csv_dir)
        if not cic_df.empty:
            combined = label_zeek_flows(combined, cic_df)
        else:
            logger.warning("No labels loaded. Assigning 'Unknown' labels.")
            combined["mapped_label"] = "Unknown"
    else:
        logger.warning("No CSV label files found in %s. Assigning 'Unknown' labels.", csv_dir)
        logger.warning("Download CIC-IDS2017 CSVs for ground-truth labels.")
        combined["mapped_label"] = "Unknown"

    combined["binary_label"] = combined["mapped_label"].apply(
        lambda x: 0 if x == "Benign" else 1
    )

    dataset_cfg = config.get("dataset", {})
    training_classes = dataset_cfg.get("training_classes", ["Benign", "DoS", "BruteForce", "PortScan"])
    excluded_classes = dataset_cfg.get("excluded_classes", ["DDoS", "Botnet", "WebAttack"])

    df_train = combined[combined["mapped_label"].isin(training_classes)].copy()
    df_zero_day = combined[combined["mapped_label"].isin(excluded_classes)].copy()

    logger.info("Training set: %d flows", len(df_train))
    logger.info("Zero-day eval set: %d flows", len(df_zero_day))
    logger.info("Training labels:\n%s", df_train["mapped_label"].value_counts().to_string())

    combined.to_csv(os.path.join(args.output_dir, "all_features_labeled.csv"), index=False)

    feature_cols = [
        c for c in combined.columns
        if c not in ("mapped_label", "binary_label", "pcap_source", "uid", "ts",
                      "src_ip", "dst_ip", "id.orig_h", "id.resp_h")
        and combined[c].dtype in (np.float64, np.int64, float, int)
    ]

    joblib.dump(feature_cols, os.path.join(args.output_dir, "feature_order.joblib"))

    from sklearn.model_selection import train_test_split

    X_all = df_train[feature_cols]
    y_binary = df_train["binary_label"]
    y_multi = df_train["mapped_label"]

    X_t1_train, X_t1_test, y_t1_train, y_t1_test = train_test_split(
        X_all, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    df_anomaly = df_train[df_train["binary_label"] == 1]
    X_anomaly = df_anomaly[feature_cols]
    y_anomaly = df_anomaly["mapped_label"]

    X_t2_train, X_t2_test, y_t2_train, y_t2_test = train_test_split(
        X_anomaly, y_anomaly, test_size=0.2, random_state=42, stratify=y_anomaly
    )

    X_t1_train.to_csv(os.path.join(args.output_dir, "tier1_X_train.csv"), index=False)
    X_t1_test.to_csv(os.path.join(args.output_dir, "tier1_X_test.csv"), index=False)
    y_t1_train.to_csv(os.path.join(args.output_dir, "tier1_y_train.csv"), index=False)
    y_t1_test.to_csv(os.path.join(args.output_dir, "tier1_y_test.csv"), index=False)

    X_t2_train.to_csv(os.path.join(args.output_dir, "tier2_X_train.csv"), index=False)
    X_t2_test.to_csv(os.path.join(args.output_dir, "tier2_X_test.csv"), index=False)
    y_t2_train.to_csv(os.path.join(args.output_dir, "tier2_y_train.csv"), index=False)
    y_t2_test.to_csv(os.path.join(args.output_dir, "tier2_y_test.csv"), index=False)

    X_zeroday = df_zero_day[feature_cols]
    y_zeroday = df_zero_day["mapped_label"]
    X_zeroday.to_csv(os.path.join(args.output_dir, "zeroday_X.csv"), index=False)
    y_zeroday.to_csv(os.path.join(args.output_dir, "zeroday_y.csv"), index=False)

    logger.info("Dataset generation complete. Files saved to %s", args.output_dir)
    logger.info("Feature count: %d", len(feature_cols))
    logger.info("Tier-1 train: %d, test: %d", len(X_t1_train), len(X_t1_test))
    logger.info("Tier-2 train: %d, test: %d", len(X_t2_train), len(X_t2_test))
    logger.info("Zero-day eval: %d", len(X_zeroday))


if __name__ == "__main__":
    main()