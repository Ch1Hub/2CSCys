import json
import logging
import argparse
import os
import sys

from src.pipeline import OfflinePipeline, LivePipeline, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log")
    ]
)
logging.getLogger("src.shap_explainer").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="IDS-ML Pipeline")
    parser.add_argument("--mode", choices=["offline", "live"], required=True,
                        help="Pipeline mode: offline (PCAP) or live (interface)")
    parser.add_argument("--config", default="config/config.json",
                        help="Path to config JSON")
    parser.add_argument("--pcap", default=None,
                        help="Path to PCAP file (offline mode)")
    parser.add_argument("--interface", default=None,
                        help="Network interface (live mode)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Duration in seconds (live mode)")
    parser.add_argument("--model-dir", default="models",
                        help="Directory containing trained models")
    parser.add_argument("--output", default=None,
                        help="Output directory for offline results")

    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    if args.mode == "offline":
        if not args.pcap:
            print("Error: --pcap required for offline mode")
            sys.exit(1)
        if not os.path.isfile(args.pcap):
            print(f"Error: PCAP not found: {args.pcap}")
            sys.exit(1)

        pipeline = OfflinePipeline(config)
        pipeline.load_models(args.model_dir)
        results = pipeline.process_pcap(args.pcap, args.output)

        print(f"\nProcessed {len(results)} records")
        mal_count = sum(1 for r in results if r["status"] == "malicious")
        ben_count = sum(1 for r in results if r["status"] == "benign")
        unk_count = sum(1 for r in results if r.get("attack") == "Unknown")
        print(f"  Benign: {ben_count}")
        print(f"  Malicious (known): {mal_count - unk_count}")
        print(f"  Malicious (unknown): {unk_count}")

        output_dir = config.get("alert", {}).get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "offline_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    elif args.mode == "live":
        if not args.interface:
            print("Error: --interface required for live mode")
            sys.exit(1)

        pipeline = LivePipeline(config)
        pipeline.load_models(args.model_dir)

        print(f"Starting live capture on {args.interface}...")
        print("Press Ctrl+C to stop.")
        pipeline.start(args.interface, args.duration)


if __name__ == "__main__":
    main()