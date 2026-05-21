#!/bin/bash
# Run all offline tests against the generated test PCAPs.
# Usage: bash scripts/run_offline_tests.sh
set -e
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PCAP_DIR="$PROJECT/data/pcaps/test_scenarios"

cd "$PROJECT"
source venv/bin/activate 2>/dev/null || true
mkdir -p output logs

run_test() {
    local name="$1" pcap="$2"
    echo ""
    echo "=============================================================="
    echo "  OFFLINE TEST: $name"
    echo "=============================================================="
    if [ ! -f "$pcap" ]; then
        echo "  SKIPPED - PCAP not found: $pcap"
        echo "  Run: bash scripts/generate_test_pcaps.sh"
        return
    fi
    timeout 50 python -m src.main --mode offline --pcap "$pcap" 2>/dev/null || true
    python3 -c "
import json
with open('output/offline_results.json') as f:
    d = json.load(f)
t = len(d)
b = sum(1 for r in d if r['status'] == 'benign')
m = t - b
u = sum(1 for r in d if r.get('attack') == 'Unknown')
a = {}
for r in d:
    k = r.get('attack', 'N/A')
    a[k] = a.get(k, 0) + 1
print(f'  TOTAL: {t} flows  |  Benign: {b}  |  Malicious: {m}  |  Unknown: {u}')
print(f'  Classes: {a}')
print()
for r in d:
    print(f'  {r[\"status\"]:<12} | {r.get(\"attack\", \"N/A\"):<16} | conf={r.get(\"confidence\", 0):.4f}')
print()
" 2>&1
}

run_test "1. BENIGN"       "$PCAP_DIR/benign_web_browsing.pcap"
run_test "2. PORTSCAN"     "$PCAP_DIR/portscan_nmap.pcap"
run_test "3. DOS"          "$PCAP_DIR/dos_synflood.pcap"
run_test "4. BRUTEFORCE"   "$PCAP_DIR/bruteforce_ssh_ftp.pcap"
run_test "5. ZERO-DAY ICMP" "$PCAP_DIR/zeroday_icmp_flood.pcap"

echo ""
echo "=============================================================="
echo "  ALL OFFLINE TESTS COMPLETE"
echo "=============================================================="