"""
Relabel the existing all_features_labeled.csv using feature-based matching.
Uses global NearestNeighbors per PCAP with normalized (dst_port, duration, total_bytes, total_pkts).
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


PCAP_TO_CSV = {
    "Monday-WorkingHours": ["Monday-WorkingHours.pcap_ISCX.csv"],
    "Tuesday-WorkingHours": ["Tuesday-WorkingHours.pcap_ISCX.csv"],
    "Wednesday-workingHours": ["Wednesday-workingHours.pcap_ISCX.csv"],
    "Friday-WorkingHours": [
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    ],
}

TRAINING_CLASSES = ["Benign", "DoS", "BruteForce", "PortScan"]
EXCLUDED_CLASSES = ["DDoS", "Botnet", "WebAttack"]


def map_label(raw_label):
    low = str(raw_label).lower().strip()
    if low in ("benign", "benvolent"):
        return "Benign"
    if "ddos" in low:
        return "DDoS"
    if "dos" in low:
        return "DoS"
    if "patator" in low or "brute" in low:
        return "BruteForce"
    if "portscan" in low or "port scan" in low:
        return "PortScan"
    if "bot" in low:
        return "Botnet"
    if "web" in low:
        return "WebAttack"
    return "Other"


def load_cic_data(csv_dir, csv_names):
    frames = []
    for name in csv_names:
        path = os.path.join(csv_dir, name)
        if not os.path.exists(path):
            logger.warning("CSV not found: %s", path)
            continue
        df = pd.read_csv(path, low_memory=False)
        df["total_bytes"] = df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]
        df["total_pkts"] = df["Total Fwd Packets"] + df["Total Backward Packets"]
        df["duration_sec"] = df["Flow Duration"] / 1_000_000.0
        df["mapped_label"] = df["Label"].apply(map_label)
        frames.append(df)
        logger.info("Loaded %s: %d flows", name, len(df))
    return pd.concat(frames, ignore_index=True)


def label_by_nn_global(zeek_df, cic_df):
    zeek_df = zeek_df.copy()
    zeek_df["total_bytes"] = zeek_df["orig_bytes"] + zeek_df["resp_bytes"]
    zeek_df["total_pkts"] = zeek_df["orig_pkts"] + zeek_df["resp_pkts"]

    z_features = ["dst_port", "duration", "total_bytes", "total_pkts"]
    c_features = ["Destination Port", "duration_sec", "total_bytes", "total_pkts"]

    z_X = zeek_df[z_features].values.astype(np.float64)
    c_X = cic_df[c_features].values.astype(np.float64)
    c_labels = cic_df["mapped_label"].values

    if len(c_X) == 0:
        return pd.Series("Benign", index=zeek_df.index)

    z_X = np.nan_to_num(z_X, nan=0.0, posinf=0.0, neginf=0.0)
    c_X = np.nan_to_num(c_X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    all_X = np.vstack([z_X, c_X])
    scaler.fit(all_X)
    z_X_norm = scaler.transform(z_X)
    c_X_norm = scaler.transform(c_X)

    logger.info("  Building BallTree on %d points...", len(c_X))
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=-1)
    nn.fit(c_X_norm)
    _, indices = nn.kneighbors(z_X_norm)

    labels = pd.Series(c_labels[indices.flatten()], index=zeek_df.index)
    return labels


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base, "data", "csv")
    output_dir = os.path.join(base, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    all_feats_path = os.path.join(output_dir, "all_features_labeled.csv")
    if not os.path.exists(all_feats_path):
        logger.error("all_features_labeled.csv not found. Run generate_dataset.py first.")
        sys.exit(1)

    logger.info("Loading all_features_labeled.csv...")
    zeek_all = pd.read_csv(all_feats_path, low_memory=False)
    logger.info("Loaded %d flows across %d PCAPs", len(zeek_all), zeek_all["pcap_source"].nunique())

    all_labeled = []

    for pcap_source, csv_names in PCAP_TO_CSV.items():
        zeek_sub = zeek_all[zeek_all["pcap_source"] == pcap_source].copy()
        if zeek_sub.empty:
            logger.warning("No Zeek flows for %s", pcap_source)
            continue

        cic_df = load_cic_data(csv_dir, csv_names)
        if cic_df.empty:
            logger.warning("No CIC data for %s, all labels = Unknown", pcap_source)
            zeek_sub["mapped_label"] = "Unknown"
            zeek_sub["binary_label"] = 1
            all_labeled.append(zeek_sub)
            continue

        logger.info("Matching %d Zeek flows to %d CIC flows for %s", len(zeek_sub), len(cic_df), pcap_source)
        import time
        t0 = time.time()
        zeek_sub["mapped_label"] = label_by_nn_global(zeek_sub, cic_df)
        elapsed = time.time() - t0
        logger.info("  Matching took %.1f seconds", elapsed)

        stats = zeek_sub["mapped_label"].value_counts()
        logger.info("Label distribution for %s:\n%s", pcap_source, stats.to_string())
        all_labeled.append(zeek_sub)

    combined = pd.concat(all_labeled, ignore_index=True)
    combined["binary_label"] = combined["mapped_label"].apply(lambda x: 0 if x == "Benign" else 1)

    total_stats = combined["mapped_label"].value_counts()
    logger.info("=== FINAL LABEL DISTRIBUTION ===")
    logger.info("\n%s", total_stats.to_string())
    logger.info("Binary (0=Benign, 1=Anomaly):\n%s", combined["binary_label"].value_counts().to_string())

    df_train = combined[combined["mapped_label"].isin(TRAINING_CLASSES)].copy()
    df_zero_day = combined[combined["mapped_label"].isin(EXCLUDED_CLASSES)].copy()

    logger.info("Training set: %d flows", len(df_train))
    logger.info("Zero-day eval set: %d flows", len(df_zero_day))
    if len(df_train) > 0:
        logger.info("Training class distribution:\n%s", df_train["mapped_label"].value_counts().to_string())
    if len(df_zero_day) > 0:
        logger.info("Zero-day class distribution:\n%s", df_zero_day["mapped_label"].value_counts().to_string())

    combined.to_csv(os.path.join(output_dir, "all_features_labeled.csv"), index=False)
    logger.info("Saved all_features_labeled.csv (%d rows)", len(combined))

    feature_cols = [
        c for c in combined.columns
        if c not in ("mapped_label", "binary_label", "pcap_source", "uid", "ts",
                      "src_ip", "dst_ip", "id.orig_h", "id.resp_h")
        and combined[c].dtype in (np.float64, np.int64, float, int)
    ]

    joblib.dump(feature_cols, os.path.join(output_dir, "feature_order.joblib"))
    logger.info("Feature count: %d", len(feature_cols))

    if len(df_train) > 0:
        X_all = df_train[feature_cols]
        y_binary = df_train["binary_label"]

        try:
            X_t1_train, X_t1_test, y_t1_train, y_t1_test = train_test_split(
                X_all, y_binary, test_size=0.2, random_state=42, stratify=y_binary
            )
        except ValueError:
            logger.warning("Stratified split failed, using random split for Tier-1")
            X_t1_train, X_t1_test, y_t1_train, y_t1_test = train_test_split(
                X_all, y_binary, test_size=0.2, random_state=42
            )

        X_t1_train.to_csv(os.path.join(output_dir, "tier1_X_train.csv"), index=False)
        X_t1_test.to_csv(os.path.join(output_dir, "tier1_X_test.csv"), index=False)
        y_t1_train.to_csv(os.path.join(output_dir, "tier1_y_train.csv"), index=False)
        y_t1_test.to_csv(os.path.join(output_dir, "tier1_y_test.csv"), index=False)

        logger.info("Tier-1 train: %d, test: %d", len(X_t1_train), len(X_t1_test))
        logger.info("Tier-1 binary - train: %s, test: %s",
                     dict(y_t1_train.value_counts()), dict(y_t1_test.value_counts()))

    df_anomaly = df_train[df_train["binary_label"] == 1]
    if len(df_anomaly) > 0:
        X_anomaly = df_anomaly[feature_cols]
        y_anomaly = df_anomaly["mapped_label"]

        class_counts = y_anomaly.value_counts()
        valid_classes = class_counts[class_counts >= 2].index.tolist()
        if len(valid_classes) < 2:
            logger.warning("Not enough Tier-2 classes for stratified split. Using random split.")
            X_t2_train, X_t2_test, y_t2_train, y_t2_test = train_test_split(
                X_anomaly, y_anomaly, test_size=0.2, random_state=42
            )
        else:
            df_anom_filt = df_anomaly[df_anomaly["mapped_label"].isin(valid_classes)]
            X_anomaly = df_anom_filt[feature_cols]
            y_anomaly = df_anom_filt["mapped_label"]
            X_t2_train, X_t2_test, y_t2_train, y_t2_test = train_test_split(
                X_anomaly, y_anomaly, test_size=0.2, random_state=42, stratify=y_anomaly
            )

        X_t2_train.to_csv(os.path.join(output_dir, "tier2_X_train.csv"), index=False)
        X_t2_test.to_csv(os.path.join(output_dir, "tier2_X_test.csv"), index=False)
        y_t2_train.to_csv(os.path.join(output_dir, "tier2_y_train.csv"), index=False)
        y_t2_test.to_csv(os.path.join(output_dir, "tier2_y_test.csv"), index=False)

        logger.info("Tier-2 train: %d, test: %d", len(X_t2_train), len(X_t2_test))
        logger.info("Tier-2 classes - train: %s, test: %s",
                     dict(y_t2_train.value_counts()), dict(y_t2_test.value_counts()))
    else:
        logger.warning("No anomalies in training set - skipping Tier-2 splits")

    if len(df_zero_day) > 0:
        X_zeroday = df_zero_day[feature_cols]
        y_zeroday = df_zero_day["mapped_label"]
        X_zeroday.to_csv(os.path.join(output_dir, "zeroday_X.csv"), index=False)
        y_zeroday.to_csv(os.path.join(output_dir, "zeroday_y.csv"), index=False)
        logger.info("Zero-day eval: %d flows", len(X_zeroday))
    else:
        logger.warning("No zero-day samples found")

    logger.info("Relabeling complete. Files saved to %s", output_dir)


if __name__ == "__main__":
    main()
