# Lightweight IDS/IPS with Zeek + Machine Learning

A two-tier intrusion detection system that combines Zeek network analysis with ensemble machine learning for real-time and offline threat detection, with built-in zero-day attack identification.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Feature Extraction](#feature-extraction)
- [Models](#models)
- [Zero-Day Detection](#zero-day-detection)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training Pipeline](#training-pipeline)
- [Alert Output Format](#alert-output-format)
- [Dataset](#dataset)

---

## Overview

This project implements a lightweight, modular Network Intrusion Detection System (NIDS) that uses Zeek for network traffic analysis and two layers of machine learning models for attack classification:

| Component | Purpose |
|---|---|
| **Zeek** | Captures and parses network traffic into structured logs |
| **Tier-1** | Binary classification — **Benign** vs **Anomaly** (LightGBM + IsolationForest) |
| **Tier-2** | Multi-class classification — **DoS**, **BruteForce**, **PortScan** (CatBoost) |
| **Unknown Detector** | Flags low-confidence predictions as **Unknown** (potential zero-days) |
| **SHAP Explainer** | Provides feature-level explanations for every alert |

The system supports two operating modes:

1. **Offline PCAP Mode** — Process saved `.pcap` files end-to-end
2. **Live Traffic Mode** — Monitor a network interface in real-time with 5s/30s sliding windows

---

## Architecture

```
                         ┌──────────────┐
                         │   Traffic    │
                         │  (PCAP/Live) │
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │     Zeek     │
                         │  (extract_   │
                         │  features.   │
                         │    zeek)     │
                         └──────┬───────┘
                                │
                    ┌───────────▼────────────┐
                    │  conn.log  dns.log      │
                    │  http.log  ssl.log      │
                    │  weird.log notice.log   │
                    └───────────┬────────────┘
                                │
                         ┌──────▼───────┐
                         │   Feature    │
                         │  Extraction  │
                         └──────┬───────┘
                                │
                    ┌───────────▼────────────┐
                    │       Tier-1            │
                    │  LightGBM + IsoForest   │
                    │  Benign / Anomaly       │
                    └──────┬───────┬─────────┘
                           │       │
                    Benign │       │ Anomaly
                           │       │
                    ┌──────▼───────▼─────────┐
                    │       Tier-2            │
                    │       CatBoost          │
                    │  DoS / BruteForce /     │
                    │  PortScan               │
                    └──────┬───────┬─────────┘
                           │       │
                  Known    │       │ Low confidence
                  attack   │       │ (< 0.65)
                           │       │
                    ┌──────▼───────▼─────────┐
                    │   Unknown Detection      │
                    │   "Unknown" classification│
                    └───────────┬────────────┘
                                │
                         ┌──────▼───────┐
                         │ SHAP Explainer│
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │ Alert Engine  │
                         │ (JSON output) │
                         └──────────────┘
```

---

## How It Works

### Offline PCAP Mode

```
PCAP file(s)
     │
     ▼
  Zeek processes PCAP → generates log files
     │
     ▼
  log_parser.py parses Zeek TSV logs into DataFrames
     │
     ▼
  feature_extractor.py builds feature vectors per connection
     │
     ▼
  Tier-1: Benign or Anomaly?
     │
     ├── Benign → output & stop
     │
     └── Anomaly → Tier-2: Which attack?
                        │
                        ├── High confidence (≥0.65) → Known attack
                        │
                        └── Low confidence (<0.65) → Unknown (potential zero-day)
     │
     ▼
  SHAP explanation + Alert JSON
```

### Live Traffic Mode

```
Network interface (e.g. eth0)
     │
     ▼
  Zeek captures traffic continuously
     │
     ▼
  window_manager.py maintains:
     - 5-second short window
     - 30-second aggregation window
     │
     ▼
  Each completed window → feature vector → Tier-1 → Tier-2 → Alert
```

---

## Feature Extraction

All features are extracted from Zeek logs (not from CICFlowMeter). This ensures the same features are used during training and inference.

### Connection Features (from `conn.log`)

| Feature | Description |
|---|---|
| `duration` | Connection duration in seconds |
| `orig_bytes` | Bytes sent by originator |
| `resp_bytes` | Bytes sent by responder |
| `orig_pkts` | Packets sent by originator |
| `resp_pkts` | Packets sent by responder |
| `service` | Service detected by Zeek |
| `conn_state` | Connection state (SF, REJ, S0, etc.) |
| `proto` | Transport protocol (tcp, udp, icmp) |

### Derived Features

| Feature | Formula |
|---|---|
| `flow_rate` | `(orig_bytes + resp_bytes) / (duration + 0.001)` |
| `bytes_ratio` | `orig_bytes / (resp_bytes + 1)` |
| `packets_ratio` | `orig_pkts / (resp_pkts + 1)` |

### DNS Features (from `dns.log`)

| Feature | Description |
|---|---|
| `dns_entropy` | Entropy of DNS query lengths |
| `nxdomain_ratio` | Ratio of NXDOMAIN responses |

### HTTP Features (from `http.log`)

| Feature | Description |
|---|---|
| `method` | HTTP request method |
| `uri_length` | URI length |
| `response_code` | HTTP response code |
| `user_agent_entropy` | Entropy of user-agent strings |

### TLS Features (from `ssl.log`)

| Feature | Description |
|---|---|
| `ja3` | JA3 fingerprint hash |
| `tls_version` | TLS/SSL version |
| `cipher_count` | Number of unique cipher suites |
| `self_signed` | Whether certificate is self-signed |

### Window Features (for live mode)

| Feature | Window | Description |
|---|---|---|
| `connections_count_5s` | 5s | Connection count in short window |
| `connections_count_30s` | 30s | Connection count in aggregation window |
| `unique_dst_ips` | 30s | Distinct destination IPs |
| `unique_dst_ports` | 30s | Distinct destination ports |
| `failed_connections` | 30s | Connections with REJ/RSTO/RSTR/S0/SH states |

---

## Models

### Tier-1: Anomaly Detection

**Goal**: Maximize recall — catch every attack, even at the cost of some false positives.

| Component | Model | Purpose |
|---|---|---|
| Primary | **LightGBM** | Binary classifier (Benign=0, Anomaly=1) |
| Secondary | **IsolationForest** | Unsupervised anomaly detector (catches what LightGBM misses) |

**Decision logic:**

```python
if (lgbm_probability > 0.30 OR iforest_detects_anomaly):
    status = "anomaly"
else:
    status = "benign"
```

The OR-combination ensures high recall (>95% target). The threshold is auto-tuned during training if the initial value doesn't meet the recall target.

### Tier-2: Attack Classification

**Goal**: Classify detected anomalies into known attack types.

| Component | Model | Purpose |
|---|---|---|
| Classifier | **CatBoost** | Multi-class: DoS, BruteForce, PortScan |

**Unknown detection logic:**

```python
probabilities = model.predict_proba(X)
max_probability = max(probabilities)

if max_probability < 0.65:
    attack = "Unknown"
```

### Training Classes vs Zero-Day Classes

| Category | Classes | Used For |
|---|---|---|
| **Training** | Benign, DoS, BruteForce, PortScan | Model training |
| **Zero-day eval** | DDoS, Botnet, WebAttack | Testing only (never seen during training) |

---

## Zero-Day Detection

The system is designed to handle previously unseen attacks:

```
DDoS traffic (never seen during training)
     │
     ▼
Tier-1: Anomaly — detected (high recall ensures this)
     │
     ▼
Tier-2: Low confidence — "I've never seen this pattern before"
     │
     ▼
Output: { "attack": "Unknown", "confidence": 0.42 }
```

The confidence threshold (0.65) is calibrated during training so that known attacks are classified confidently (>0.65) while zero-day attacks fall below the threshold.

---

## Project Structure

```
2CSCys/
├── config/
│   └── config.json                  # All configuration (thresholds, paths, features)
├── zeek/
│   └── extract_features.zeek        # Zeek script for log extraction
├── src/
│   ├── __init__.py
│   ├── main.py                      # CLI entry point (offline/live modes)
│   ├── zeek_runner.py               # Runs Zeek on PCAPs or live interfaces
│   ├── log_parser.py                # Parses Zeek TSV logs into DataFrames
│   ├── feature_extractor.py         # Builds feature vectors from Zeek logs
│   ├── window_manager.py            # 5s/30s sliding windows for live mode
│   ├── tier1.py                     # LightGBM + IsolationForest inference
│   ├── tier2.py                     # CatBoost inference
│   ├── unknown_detector.py          # Confidence threshold check
│   ├── alert_engine.py              # Formats and saves alert JSONs
│   ├── shap_explainer.py            # SHAP explanations for both tiers
│   └── pipeline.py                  # OfflinePipeline and LivePipeline orchestration
├── notebooks/
│   ├── 01_dataset_preparation.ipynb # PCAP → Zeek → features + CSV labels
│   ├── 02_tier1_training.ipynb      # Train LightGBM + IsolationForest
│   └── 03_tier2_training.ipynb      # Train CatBoost + zero-day evaluation
├── data/
│   ├── pcaps/                       # CIC-IDS2017 PCAP files (place here)
│   ├── csv/                         # CIC-IDS2017 CSV label files (place here)
│   └── processed/                   # Generated training datasets
├── models/                          # Saved models, preprocessors, thresholds
├── logs/                            # Zeek logs and alert logs
├── output/                          # Alert JSON output files
├── generate_dataset.py              # CLI script to generate dataset from PCAPs
├── requirements.txt                 # Python dependencies
├── setup.sh                         # Environment setup script
└── .gitignore
```

---

## Setup

### Prerequisites

- **Python 3.10+**
- **Zeek** — for PCAP processing and live capture
- **CIC-IDS2017 dataset** — PCAP files + CSV label files

### Install Zeek

```bash
# Ubuntu/Debian
sudo apt install zeek

# macOS
brew install zeek

# Verify
zeek --version
```

### Install Python Dependencies

```bash
bash setup.sh
```

This creates a virtual environment, installs all dependencies, creates directory structure, and checks for Zeek.

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### 1. Generate Training Dataset from PCAPs

Place your CIC-IDS2017 files:
- **PCAPs** in `data/pcaps/`
- **CSVs** (for labels only) in `data/csv/`

```bash
source venv/bin/activate
python generate_dataset.py --pcap-dir data/pcaps --csv-dir data/csv --output-dir data/processed
```

Or use the notebook: `notebooks/01_dataset_preparation.ipynb`

This processes each PCAP through Zeek, extracts features, and assigns labels by matching source/destination IPs from the CSV files.

### 2. Train Models

Run the notebooks in order:

```bash
jupyter notebook
# Open notebooks/02_tier1_training.ipynb → Run All
# Open notebooks/03_tier2_training.ipynb → Run All
```

Saved artifacts in `models/`:

| File | Description |
|---|---|
| `tier1_lgbm.joblib` | LightGBM binary classifier |
| `tier1_iforest.joblib` | IsolationForest anomaly detector |
| `tier1_preprocessor.joblib` | StandardScaler for Tier-1 |
| `tier1_feature_order.joblib` | Feature column order |
| `tier1_thresholds.joblib` | LGBM threshold, target recall |
| `tier1_shap_explainer.joblib` | SHAP explainer for Tier-1 |
| `tier2_catboost.joblib` | CatBoost multi-class classifier |
| `tier2_preprocessor.joblib` | StandardScaler for Tier-2 |
| `tier2_label_encoder.joblib` | Label encoder for attack classes |
| `tier2_feature_order.joblib` | Feature column order |
| `tier2_thresholds.joblib` | Unknown threshold, class list |
| `tier2_shap_explainer.joblib` | SHAP explainer for Tier-2 |

### 3. Run Inference — Offline PCAP Mode

```bash
source venv/bin/activate
python -m src.main --mode offline --pcap /path/to/file.pcap
```

### 4. Run Inference — Live Traffic Mode

```bash
source venv/bin/activate
python -m src.main --mode live --interface eth0 --duration 300
```

Without `--duration`, it runs until Ctrl+C.

---

## Configuration

All settings are in `config/config.json`:

```json
{
    "zeek": {
        "binary_path": "zeek",
        "output_dir": "logs",
        "scripts_dir": "zeek"
    },
    "tier1": {
        "model_type": "LightGBM",
        "secondary_model": "IsolationForest",
        "lgbm_threshold": 0.30,
        "target_recall": 0.95
    },
    "tier2": {
        "model_type": "CatBoost",
        "classes": ["DoS", "BruteForce", "PortScan"],
        "unknown_threshold": 0.65
    },
    "window": {
        "short_window_seconds": 5,
        "aggregation_window_seconds": 30
    },
    "dataset": {
        "name": "CIC-IDS2017",
        "training_classes": ["Benign", "DoS", "BruteForce", "PortScan"],
        "excluded_classes": ["DDoS", "Botnet", "WebAttack"],
        "primary_unknown": "DDoS"
    }
}
```

The `lgbm_threshold` is auto-tuned during Tier-1 training if the initial value doesn't achieve 95% recall. The updated threshold is written back to the config file.

---

## Training Pipeline

### Notebook 01 — Dataset Generation

- Processes each PCAP through **Zeek** to produce connection, DNS, HTTP, and TLS logs
- Parses logs with `log_parser.py` and extracts features with `feature_extractor.py`
- Loads CIC-IDS2017 CSV files **for labels only** (matching by source/destination IP)
- Maps labels to training classes (Benign, DoS, BruteForce, PortScan) and zero-day classes (DDoS, Botnet, WebAttack)
- Creates stratified train/test splits for both Tier-1 and Tier-2
- Saves processed datasets to `data/processed/`

### Notebook 02 — Tier-1 Training

- Trains **LightGBM** binary classifier (Benign vs Anomaly) with class balancing via `scale_pos_weight`
- Trains **IsolationForest** as secondary anomaly detector
- Evaluates combined decision logic: `Anomaly if (LGBM_prob > 0.30 OR IsoForest flags)`
- Auto-tunes LGBM threshold if recall < 95%
- Generates SHAP feature importance plot
- Saves all models and artifacts to `models/`

### Notebook 03 — Tier-2 Training

- Trains **CatBoost** multi-class classifier on anomaly samples only (DoS, BruteForce, PortScan)
- Evaluates classification metrics (Precision, Recall, F1 per class)
- Tests unknown detection threshold on zero-day data (DDoS, Botnet, WebAttack)
- Runs full end-to-end pipeline test: Tier-1 → Tier-2 → Unknown detection on zero-day samples
- Generates SHAP feature importance plot
- Saves all models and artifacts to `models/`

---

## Alert Output Format

Every alert follows this JSON schema:

```json
{
    "status": "malicious",
    "attack": "Unknown",
    "confidence": 0.45,
    "source": "pcap",
    "window_id": 123,
    "explanation": {
        "top_features": [
            {
                "feature": "connections_count_30s",
                "importance": 0.32,
                "direction": "positive",
                "shap_value": 0.28
            }
        ]
    },
    "timestamp": "2026-05-20T14:30:00.000Z",
    "tier1_detail": {
        "status": "anomaly",
        "probability": 0.87
    },
    "tier2_detail": {
        "attack": "Unknown",
        "confidence": 0.45,
        "all_probabilities": {
            "DoS": 0.18,
            "BruteForce": 0.12,
            "PortScan": 0.15
        }
    },
    "unknown_detection": {
        "is_unknown": true,
        "attack": "Unknown",
        "confidence": 0.45,
        "reason": "Low confidence (0.45 < 0.65)"
    }
}
```

---

## Dataset

**CIC-IDS2017** — Canadian Institute for Cybersecurity Intrusion Detection System 2017 dataset.

Download from: https://www.unb.ca/cic/datasets/ids-2017.html

### What you need

| File Type | Location | Purpose |
|---|---|---|
| PCAP files (`.pcap`) | `data/pcaps/` | Processed through Zeek for feature extraction |
| CSV files (`.csv`) | `data/csv/` | Used **only for ground-truth labels** (not for features) |

### Why PCAPs and not CSVs directly?

The CIC-IDS2017 CSVs contain 80+ features computed by CICFlowMeter (bytes/sec, packet length statistics, IAT distributions, etc.). These features are **not the same** as what Zeek extracts from live traffic. If we trained on CICFlowMeter features, the models would fail at inference time because the feature distributions would be completely different.

Instead, the correct approach is:

1. Process PCAPs through **our Zeek script** → extract the exact same features the system will use at runtime
2. Use CSV files **only** to look up ground-truth labels by matching source/destination IP addresses
3. Train models on **Zeek features + CSV labels**

This guarantees training-inference feature consistency.

### Label Mapping

| CIC-IDS2017 Label | Mapped To | Category |
|---|---|---|
| BENIGN | Benign | Training |
| DoS slowloris, DoS Slowhttptest, DoS Hulk, DoS GoldenEye | DoS | Training |
| FTP-Patator, SSH-Patator | BruteForce | Training |
| PortScan | PortScan | Training |
| DDoS | DDoS | Zero-day eval |
| Bot | Botnet | Zero-day eval |
| Web Attack – Brute Force, XSS, SQL Injection | WebAttack | Zero-day eval |