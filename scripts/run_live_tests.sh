#!/bin/bash
# Run live traffic detection tests on loopback/enp1s0 interfaces.
# Usage: bash scripts/run_live_tests.sh
set -e
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"
source venv/bin/activate 2>/dev/null || true

run_live() {
    local name="$1" iface="$2" dur="$3"
    local dir="/tmp/live_${name}_$$"
    shift 3
    echo ""
    echo "=============================================================="
    echo "  LIVE TEST: $name (${iface}, ${dur}s)"
    echo "=============================================================="
    rm -rf "$dir" 2>/dev/null; mkdir -p "$dir/output"
    python3 -c "
import json
c = json.load(open('config/config.json'))
c['zeek']['output_dir'] = '$dir'
c['alert']['output_dir'] = '$dir/output'
c['alert']['log_dir'] = '$dir'
json.dump(c, open('$dir/config.json', 'w'), indent=4)
"
    timeout $((dur + 20)) python -m src.main --mode live --interface "$iface" --duration "$dur" \
        --config "$dir/config.json" --model-dir models > "$dir/ids.log" 2>&1 &
    local PID=$!
    sleep 8
    echo "  [traffic] Generating..."
    "$@" 2>/dev/null || true
    wait $PID 2>/dev/null || true
    sleep 3

    local alerts="$dir/alerts.jsonl"
    if [ -f "$alerts" ]; then
        python3 -c "
import json
with open('$alerts') as f:
    lines = [l.strip() for l in f if l.strip().startswith('{')]
items = [json.loads(l) for l in lines]
t = len(items)
b = sum(1 for r in items if r['status'] == 'benign')
m = t - b
u = sum(1 for r in items if r.get('attack') == 'Unknown')
a = {}
for r in items:
    k = r.get('attack', 'N/A')
    a[k] = a.get(k, 0) + 1
pct = b / t * 100 if t > 0 else 0
print(f'  Alerts: {t}  |  Benign: {b} ({pct:.0f}%)  |  Malicious: {m}  |  Unknown: {u}')
print(f'  Classes: {a}')
for r in items[:3]:
    print(f'    {r[\"status\"]:<12} | {r.get(\"attack\", \"N/A\"):<16} | conf={r.get(\"confidence\", 0):.4f}')
if len(items) > 3:
    print(f'    ... +{len(items) - 3} more')
"
    else
        echo "  No alerts generated"
    fi
    sleep 2
}

# 1 - BENIGN (enp1s0, 30s)
run_live "BENIGN" "enp1s0" 30 bash -c '
    curl -s http://httpbin.org/get -o /dev/null
    curl -s https://httpbin.org/ip -o /dev/null
    for h in google.com github.com; do host "$h" 2>/dev/null; done
    ping -c 3 -W 1 8.8.8.8
'

# 2 - PORTSCAN (lo, 30s)
run_live "PORTSCAN" "lo" 30 bash -c '
    nmap -T4 -p 1-500 --max-retries 0 127.0.0.1
'

# 3 - DOS (lo, 30s)
run_live "DOS" "lo" 30 bash -c '
    timeout 3 hping3 -S -p 80 --faster -c 15 127.0.0.1
'

# 4 - BRUTEFORCE (lo, 45s)
run_live "BRUTEFORCE" "lo" 45 bash -c '
python3 -c "
import socket,time,threading,random
def bf(p,d):
 try:
  s=socket.socket();s.settimeout(d+2);s.connect((\"127.0.0.1\",p))
  t=time.time()
  while time.time()-t<d:
   try:s.send(b\"A\"*random.randint(40,200))
   except:break
   time.sleep(random.uniform(0.3,1.0))
  s.close()
 except:pass
ts=[]
for i in range(5):
 t=threading.Thread(target=bf,args=(random.choice([21,22]),random.uniform(8,12)))
 t.start();ts.append(t);time.sleep(0.3)
for t in ts:t.join(15)
"
'

echo ""
echo "=============================================================="
echo "  ALL LIVE TESTS COMPLETE"
echo "=============================================================="