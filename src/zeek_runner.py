import subprocess
import logging
import os
import shutil
import signal
import time
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ZeekRunner:
    def __init__(self, config: dict):
        self.zeek_binary = config.get("zeek", {}).get("binary_path", "zeek")
        self.scripts_dir = os.path.join(PROJECT_ROOT, config.get("zeek", {}).get("scripts_dir", "zeek"))
        self.output_dir = config.get("zeek", {}).get("output_dir", "logs")
        os.makedirs(self.output_dir, exist_ok=True)

    def run_offline(self, pcap_path: str, output_dir: Optional[str] = None) -> str:
        pcap_path = os.path.abspath(pcap_path)
        if not os.path.isfile(pcap_path):
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

        out_dir = output_dir or self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        script_path = os.path.join(self.scripts_dir, "extract_features.zeek")
        cmd = [
            self.zeek_binary,
            "-r", pcap_path,
            "-C",
            script_path
        ]

        logger.info("Running Zeek offline on %s", pcap_path)
        result = subprocess.run(
            cmd,
            cwd=out_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error("Zeek stderr: %s", result.stderr)
            raise RuntimeError(f"Zeek exited with code {result.returncode}: {result.stderr}")

        log_files = [f for f in os.listdir(out_dir) if f.endswith(".log")]
        logger.info("Zeek produced %d log files: %s", len(log_files), log_files)
        return out_dir

    def run_live(self, interface: str, output_dir: Optional[str] = None,
                 duration: Optional[int] = None) -> subprocess.Popen:
        out_dir = output_dir or self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        script_path = os.path.join(self.scripts_dir, "extract_features.zeek")
        cmd = [
            self.zeek_binary,
            "-i", interface,
            "-C",
            script_path
        ]

        logger.info("Starting Zeek live capture on interface %s", interface)
        proc = subprocess.Popen(
            cmd,
            cwd=out_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        if duration:
            logger.info("Live capture will run for %d seconds", duration)
            time.sleep(duration)
            self.stop_live(proc)
            return proc

        return proc

    @staticmethod
    def stop_live(proc: subprocess.Popen):
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                logger.info("Zeek live capture stopped")
            except ProcessLookupError:
                pass

    def get_log_paths(self, output_dir: str) -> dict:
        expected = ["conn", "dns", "http", "ssl", "weird", "notice"]
        paths = {}
        for key in expected:
            for suffix in ["", "-2", "-3", "-4"]:
                filename = f"{key}.log" if not suffix else f"{key}{suffix}.log"
                path = os.path.join(output_dir, filename)
                if os.path.isfile(path):
                    paths[key] = path
                    break
            if key not in paths:
                logger.debug("Expected log not found: %s/%s.log", output_dir, key)
        return paths