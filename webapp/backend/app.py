"""
Flask REST API for 2CSCys IDS Web Interface.
"""

import json
import os
import sys
import uuid
import threading
import subprocess
import time
import signal
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import OfflinePipeline, load_config
from src.alert_engine import AlertEngine

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

config = load_config(os.path.join(PROJECT_ROOT, "config", "config.json"))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
ALERTS_FILE = os.path.join(LOG_DIR, "alerts.jsonl")

jobs = {}
live_proc = None
live_interface = None


def _run_offline(job_id, pcap_path):
    try:
        pipeline = OfflinePipeline(config)
        pipeline.load_models(MODEL_DIR)
        results = pipeline.process_pcap(pcap_path)

        total = len(results)
        benign = sum(1 for r in results if r["status"] == "benign")
        malicious = sum(1 for r in results if r["status"] == "malicious")
        unknown = sum(1 for r in results if r.get("attack") == "Unknown")

        # Write to alerts.jsonl
        ae = AlertEngine(config)
        for r in results:
            ae.log_alert(r)

        jobs[job_id] = {
            "status": "completed",
            "filename": os.path.basename(pcap_path),
            "summary": {"total": total, "benign": benign, "malicious": malicious, "unknown": unknown},
            "results": results,
        }
    except Exception as e:
        jobs[job_id] = {"status": "failed", "filename": os.path.basename(pcap_path), "error": str(e)}
    finally:
        if os.path.exists(pcap_path):
            os.remove(pcap_path)


# ── Offline ──────────────────────────────────────────────

@app.route("/api/offline/upload", methods=["POST"])
def offline_upload():
    if "pcap" not in request.files:
        return jsonify({"error": "No PCAP file provided"}), 400
    file = request.files["pcap"]
    if not file.filename.endswith((".pcap", ".pcapng")):
        return jsonify({"error": "Invalid file type. Use .pcap or .pcapng"}), 400

    job_id = str(uuid.uuid4())[:8]
    pcap_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    file.save(pcap_path)

    jobs[job_id] = {"status": "processing", "filename": file.filename}
    t = threading.Thread(target=_run_offline, args=(job_id, pcap_path), daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "status": "processing"})


@app.route("/api/offline/status/<job_id>", methods=["GET"])
def offline_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"job_id": job_id, "status": job["status"],
                    "summary": job.get("summary")})


@app.route("/api/offline/results/<job_id>", methods=["GET"])
def offline_results(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "Job not completed", "status": job["status"]}), 400
    return jsonify({"job_id": job_id, "summary": job["summary"], "results": job["results"]})


@app.route("/api/offline/list", methods=["GET"])
def offline_list():
    return jsonify([{"job_id": k, "status": v["status"], "filename": v.get("filename", ""),
                     "summary": v.get("summary")} for k, v in jobs.items()])


# ── Live ─────────────────────────────────────────────────

@app.route("/api/live/start", methods=["POST"])
def live_start():
    global live_proc, live_interface
    data = request.get_json() or {}
    interface = data.get("interface", "lo")

    if live_proc and live_proc.poll() is None:
        return jsonify({"error": "Live capture already running"}), 400

    venv_python = os.path.join(PROJECT_ROOT, "venv", "bin", "python")
    cmd = [
        venv_python, "-m", "src.main",
        "--mode", "live",
        "--interface", interface,
        "--config", os.path.join(PROJECT_ROOT, "config", "config.json"),
        "--model-dir", MODEL_DIR,
    ]
    live_proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    live_interface = interface

    return jsonify({"status": "started", "interface": interface, "pid": live_proc.pid})


@app.route("/api/live/stop", methods=["POST"])
def live_stop():
    global live_proc, live_interface
    if live_proc and live_proc.poll() is None:
        live_proc.send_signal(signal.SIGINT)
        try:
            live_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            live_proc.terminate()
        live_proc = None
        live_interface = None
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not running"})


@app.route("/api/live/status", methods=["GET"])
def live_status():
    running = live_proc is not None and live_proc.poll() is None
    return jsonify({"running": running, "pid": live_proc.pid if running else None,
                    "interface": live_interface})


# ── Alerts ───────────────────────────────────────────────

def _read_alerts():
    alerts = []
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    alerts.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return alerts


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    attack_filter = request.args.get("attack", "").strip().lower()
    status_filter = request.args.get("status", "").strip().lower()
    search = request.args.get("search", "").strip().lower()

    alerts = _read_alerts()

    if status_filter:
        alerts = [a for a in alerts if a.get("status", "").lower() == status_filter]
    if attack_filter:
        alerts = [a for a in alerts if a.get("attack", "").lower() == attack_filter]
    if search:
        alerts = [a for a in alerts if search in json.dumps(a).lower()]

    alerts.reverse()
    total = len(alerts)
    start = (page - 1) * per_page
    end = start + per_page

    return jsonify({
        "total": total, "page": page, "per_page": per_page,
        "pages": max((total + per_page - 1) // per_page, 1),
        "alerts": alerts[start:end],
    })


@app.route("/api/alerts/stats", methods=["GET"])
def alerts_stats():
    alerts = _read_alerts()
    total = len(alerts)
    benign = sum(1 for a in alerts if a.get("status") == "benign")
    malicious = total - benign
    unknown = sum(1 for a in alerts if a.get("attack") == "Unknown")
    by_attack = {}
    for a in alerts:
        atk = a.get("attack", "N/A")
        by_attack[atk] = by_attack.get(atk, 0) + 1

    return jsonify({
        "total": total, "benign": benign, "malicious": malicious,
        "unknown": unknown, "by_attack": by_attack,
    })


# ── Models ───────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def model_info():
    files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")])
    pngs = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".png")])
    return jsonify({
        "tier1": {"model": "LightGBM", "threshold": config.get("tier1", {}).get("lgbm_threshold_nof", 0.50),
                   "use_iforest": config.get("tier1", {}).get("use_iforest", False)},
        "tier2": {"model": "CatBoost", "classes": config.get("tier2", {}).get("classes", []),
                   "unknown_threshold": config.get("tier2", {}).get("unknown_threshold", 0.65)},
        "files": files, "shap_plots": pngs,
    })


@app.route("/api/models/shap/<tier>", methods=["GET"])
def shap_plot(tier):
    if tier not in ("tier1", "tier2"):
        return jsonify({"error": "Invalid tier"}), 400
    path = os.path.join(MODEL_DIR, f"{tier}_shap_summary.png")
    if os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return jsonify({"error": "SHAP plot not found"}), 404


# ── Health ───────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})


# ── Serve React SPA frontend ─────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")

_NO_SPA = not os.path.isfile(os.path.join(STATIC_DIR, "index.html"))


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(STATIC_DIR, "assets"), filename)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    if path.startswith("api/"):
        from flask import abort
        abort(404)
    if _NO_SPA:
        return jsonify({"error": "Frontend not built. Run: cd webapp/frontend && npm install && npx vite build"}), 503
    return send_file(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    print(f"2CSCys Dashboard → http://{hostname}:5000")
    print(f"                      http://10.0.145.2:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
