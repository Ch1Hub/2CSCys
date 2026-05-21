# Lightweight NIDS with Zeek + Machine Learning

A two-tier Network Intrusion Detection System combining **Zeek** network analysis with **LightGBM + CatBoost** ensemble for real-time and offline threat detection, with built-in zero-day attack identification.

---

## Architecture

```
Traffic (PCAP / Live)
       │
       ▼
     Zeek ─── conn.log, dns.log, http.log, ssl.log
       │
       ▼
  Feature Extraction (22 features per flow)
       │
       ▼
  ┌─ Tier-1: LightGBM ──► Benign or Anomaly? ──┐
  │                                             │
  │  Benign → stop                              │
  │  Anomaly → Tier-2                           │
  │                                             │
  └─ Tier-2: CatBoost ──► DoS / BruteForce / PortScan
                    │
                    ├── confidence ≥ 0.65 → Known attack
                    └── confidence < 0.65 → Unknown (zero-day)
       │
       ▼
  Alert Engine → JSON output
```

### Two Operating Modes

| Mode | Input | Processing | Output |
|---|---|---|---|
| **Offline** | PCAP file | Batch processing | `output/offline_results.json` |
| **Live** | Network interface | 5s/30s sliding windows | `logs/alerts.jsonl` |

---

## Project Structure

```
2CSCys/
├── config/
│   └── config.json                 # Thresholds, paths, feature config
├── zeek/
│   └── extract_features.zeek       # Zeek log extraction script
├── src/
│   ├── main.py                     # CLI entry point
│   ├── pipeline.py                 # OfflinePipeline + LivePipeline
│   ├── zeek_runner.py              # Zeek process management
│   ├── log_parser.py               # Zeek TSV → DataFrames
│   ├── feature_extractor.py        # 22 features from Zeek logs
│   ├── window_manager.py           # 5s/30s sliding windows (live)
│   ├── tier1.py                    # LightGBM binary classifier
│   ├── tier2.py                    # CatBoost multi-class classifier
│   ├── unknown_detector.py         # Confidence < 0.65 = Unknown
│   ├── alert_engine.py             # JSON alert generation
│   └── shap_explainer.py           # SHAP feature importance
├── scripts/
│   ├── generate_dataset.py         # PCAPs → Zeek → features → labels
│   ├── relabel_dataset.py          # Feature-based fuzzy label matching
│   ├── train_tier1.py              # Train LightGBM
│   ├── train_tier2.py              # Train CatBoost
│   ├── generate_test_pcaps.sh      # Create test scenario PCAPs
│   ├── run_offline_tests.sh        # Run all offline tests
│   └── run_live_tests.sh           # Run all live tests
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_tier1_training.ipynb
│   └── 03_tier2_training.ipynb
├── data/
│   ├── pcaps/                      # Training PCAPs + test_scenarios/
│   ├── csv/                        # CIC-IDS2017 label CSVs
│   └── processed/                  # Generated training datasets
├── models/                         # 12 trained model artifacts
├── output/                         # Offline pipeline results
├── logs/                           # Live pipeline alerts
└── requirements.txt
```

---

## Models

### Tier-1: Anomaly Detection

| Component | Model | Threshold | Performance |
|---|---|---|---|
| Primary | **LightGBM** | anomaly_prob > 0.50 | 99.6% recall, 0.5% FPR |

LightGBM binary classifier trained on 1.2M flows from CIC-IDS2017. The 0.50 threshold balances high recall with low false positive rate.

### Tier-2: Attack Classification

| Component | Model | Classes | Performance |
|---|---|---|---|
| Classifier | **CatBoost** | DoS, BruteForce, PortScan | 99.97% weighted F1 |

Confidence < 0.65 triggers **Unknown** classification for potential zero-day attacks.

### Features (22 total)

| Category | Features |
|---|---|
| Connection | `duration`, `orig_bytes`, `resp_bytes`, `orig_pkts`, `resp_pkts`, `dst_port`, `service`, `conn_state`, `proto` |
| Derived | `flow_rate`, `bytes_ratio`, `packets_ratio` |
| DNS | `dns_entropy`, `nxdomain_ratio` |
| HTTP | `uri_length`, `response_code`, `user_agent_entropy` |
| TLS | `ja3_hash`, `cipher_count`, `self_signed` |
| Window | `connections_count_5s`, `connections_count_30s`, `unique_dst_ips`, `unique_dst_ports`, `failed_connections` |

---

## Setup

### Prerequisites

- **Python 3.10+**
- **Zeek 4.0+** (`sudo apt install zeek`)
- Test tools: `nmap`, `hping3`, `hydra` (for live attack generation)

### Install

```bash
git clone https://github.com/Ch1Hub/Lightweight-NIDS-with-Zeek-Machine-Learning.git
cd 2CSCys
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dataset

Download CIC-IDS2017 from https://www.unb.ca/cic/datasets/ids-2017.html:
- Place **PCAP files** in `data/pcaps/`
- Place **CSV label files** in `data/csv/`

---

## Training Pipeline

### 1. Generate Dataset

```bash
source venv/bin/activate
python scripts/generate_dataset.py --pcap-dir data/pcaps --csv-dir data/csv --output-dir data/processed
```

Processes each PCAP through Zeek, extracts 22 features per flow, and matches labels from CIC-IDS2017 CSVs (by IP or feature-based fuzzy matching).

### 2. Train Tier-1 (LightGBM)

```bash
python scripts/train_tier1.py
```

Saves: `tier1_lgbm.joblib`, `tier1_preprocessor.joblib`, `tier1_feature_order.joblib`, `tier1_thresholds.joblib`, `tier1_shap_explainer.joblib`

### 3. Train Tier-2 (CatBoost)

```bash
python scripts/train_tier2.py
```

Saves: `tier2_catboost.joblib`, `tier2_preprocessor.joblib`, `tier2_label_encoder.joblib`, `tier2_feature_order.joblib`, `tier2_thresholds.joblib`, `tier2_shap_explainer.joblib`

---

## Running the IDS

### Offline Mode (PCAP analysis)

```bash
source venv/bin/activate
python -m src.main --mode offline --pcap /path/to/capture.pcap
```

### Live Mode (real-time monitoring)

```bash
source venv/bin/activate
python -m src.main --mode live --interface eth0 --duration 300
```

Without `--duration`, runs until Ctrl+C.

### Running Tests

```bash
# Generate test PCAPs
bash scripts/generate_test_pcaps.sh

# Run all offline tests
bash scripts/run_offline_tests.sh

# Run all live tests
bash scripts/run_live_tests.sh
```

---

## Alert Format

```json
{
    "status": "malicious",
    "attack": "PortScan",
    "confidence": 0.97,
    "source": "pcap",
    "window_id": 42,
    "tier1_detail": { "status": "anomaly", "probability": 0.98 },
    "tier2_detail": {
        "attack": "PortScan",
        "confidence": 0.97,
        "all_probabilities": { "DoS": 0.01, "BruteForce": 0.02, "PortScan": 0.97 }
    },
    "unknown_detection": { "is_unknown": false },
    "explanation": { "top_features": [...] },
    "timestamp": "2026-05-21T12:00:00.000Z"
}
```

---

## Test Results

### Offline (PCAP files)

| Test | Flows | Detection | Tier-2 |
|---|---|---|---|
| Benign (web) | 7 | 6/7 (86%) | 1 Unk, 6 N/A |
| PortScan (nmap) | 49 | 48/49 (98%) | 46 PortScan |
| DoS (SYN flood) | 10 | 10/10 (100%) | 10 DoS |
| BruteForce (SSH/FTP) | 8 | 8/8 (100%) | 8 BruteForce |
| Zero-day (ICMP flood) | 1 | 1/1 (100%) | 1 Unknown |

### Live (interface capture)

| Test | Alerts | Result | Tier-2 |
|---|---|---|---|
| Benign (web) | 8 | 100% benign | All N/A |
| PortScan (nmap) | 500 | 96% malicious | 481 PortScan |
| DoS (SYN flood) | 15 | 100% DoS | 15 DoS |
| BruteForce (SSH/FTP) | 4 | 100% BruteForce | 4 BruteForce |

---

## Configuration

Key settings in `config/config.json`:

```json
{
    "tier1": {
        "model_type": "LightGBM",
        "lgbm_threshold": 0.30,
        "lgbm_threshold_nof": 0.50,
        "use_iforest": false
    },
    "tier2": {
        "model_type": "CatBoost",
        "classes": ["BruteForce", "DoS", "PortScan"],
        "unknown_threshold": 0.65
    },
    "window": {
        "short_window_seconds": 5,
        "aggregation_window_seconds": 30
    }
}
```

---

## Dataset

**CIC-IDS2017** — https://www.unb.ca/cic/datasets/ids-2017.html

| Label | Mapped To | Category |
|---|---|---|
| BENIGN | Benign | Training |
| DoS Hulk, GoldenEye, Slowloris, Slowhttptest | DoS | Training |
| FTP-Patator, SSH-Patator | BruteForce | Training |
| PortScan | PortScan | Training |
| DDoS, Bot, Web Attack | Excluded | Zero-day eval |

Features are extracted from PCAPs via Zeek at both training and inference time — the CIC-IDS2017 CSVs provide ground-truth labels only, never features.

---

## Known Limitations

- **Application-layer attacks** (SQLi, XSS) are invisible to flow-level features — the model only sees connection metadata
- **BruteForce detection** requires long-duration connections (8-12s) matching CIC-IDS2017 patterns; rapid SSH scans on loopback may be missed
- **Zero-day DDoS** shares per-flow features with DoS — Tier-2 classifies it as DoS with high confidence
- **Live ICMP capture** is limited — ICMP floods produce few conn.log entries in live mode

## Dependencies

```
lightgbm>=4.0      catboost>=1.2      scikit-learn>=1.3
pandas>=2.0        numpy>=1.24        joblib>=1.3
shap>=0.42         matplotlib>=3.7    pyarrow>=14.0
```

## Labeling Approach

The CIC-IDS2017 CSV files provide ground-truth labels. Two matching strategies are supported:

| Strategy | When Used | Method |
|---|---|---|
| **IP-based** | CSV has Source IP + Destination IP columns | Direct (src_ip, dst_ip) → label lookup |
| **Feature-based** | IP columns missing (parquet-converted CSVs) | NearestNeighbor on (dst_port, duration, bytes, pkts) |

The `scripts/relabel_dataset.py` script handles the feature-based fuzzy matching when the original IP-labeled CSVs are unavailable.

---

## Author

**Ch1Hub** — yc.chikhaoui@esi-sba.dz

École Nationale Supérieure d'Informatique (ESI), Sidi Bel Abbès, Algeria

---

## License

This project is developed for academic research purposes. CIC-IDS2017 dataset is property of the Canadian Institute for Cybersecurity.
