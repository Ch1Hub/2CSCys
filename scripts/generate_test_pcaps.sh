#!/bin/bash
# Generate all test scenario PCAPs for offline testing.
# Run from project root.
set -e
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$PROJECT/data/pcaps/test_scenarios"
mkdir -p "$OUT"

cd "$PROJECT"
source venv/bin/activate 2>/dev/null || true

echo "=== Generating Test PCAPs in $OUT ==="

# 1 - BENIGN (web browsing)
echo "[1/5] Benign web browsing..."
tcpdump -i enp1s0 -w "$OUT/benign_web_browsing.pcap" -c 80 &>/dev/null &
PID=$!; sleep 1
curl -s http://httpbin.org/get -o /dev/null 2>/dev/null || true
curl -s https://httpbin.org/ip -o /dev/null 2>/dev/null || true
for h in google.com github.com; do timeout 2 host "$h" 2>/dev/null; done
ping -c 1 -W 1 8.8.8.8 2>/dev/null || true
sleep 2; kill $PID 2>/dev/null; wait $PID 2>/dev/null || true
echo "   -> $(ls -lh "$OUT/benign_web_browsing.pcap" | awk '{print $5}')"

# 2 - PORTSCAN (nmap)
echo "[2/5] PortScan (nmap)..."
tcpdump -i lo -w "$OUT/portscan_nmap.pcap" -c 100 &>/dev/null &
PID=$!; sleep 0.5
nmap -T4 -p 1-500 --max-retries 0 127.0.0.1 2>/dev/null || true
sleep 1; kill $PID 2>/dev/null; wait $PID 2>/dev/null || true
echo "   -> $(ls -lh "$OUT/portscan_nmap.pcap" | awk '{print $5}')"

# 3 - DOS (SYN flood)
echo "[3/5] DoS SYN flood..."
tcpdump -i lo -w "$OUT/dos_synflood.pcap" -c 30 &>/dev/null &
PID=$!; sleep 0.5
timeout 2 hping3 -S -p 80 --faster -c 10 127.0.0.1 2>/dev/null || true
sleep 1; kill $PID 2>/dev/null; wait $PID 2>/dev/null || true
echo "   -> $(ls -lh "$OUT/dos_synflood.pcap" | awk '{print $5}')"

# 4 - BRUTEFORCE (slow SSH/FTP)
echo "[4/5] BruteForce (slow SSH/FTP)..."
tcpdump -i lo -w "$OUT/bruteforce_ssh_ftp.pcap" -c 400 &>/dev/null &
PID=$!; sleep 0.5
python3 -c "
import socket,time,threading,random
def bf(p,d):
 try:
  s=socket.socket();s.settimeout(d+2);s.connect(('127.0.0.1',p))
  t=time.time()
  while time.time()-t<d:
   try:s.send(b'A'*random.randint(40,200))
   except:break
   time.sleep(random.uniform(0.3,1.0))
  s.close()
 except:pass
ts=[]
for i in range(5):
 t=threading.Thread(target=bf,args=(random.choice([21,22]),random.uniform(8,12)))
 t.start();ts.append(t);time.sleep(0.3)
for t in ts:t.join(15)
" 2>/dev/null || true
sleep 3; kill $PID 2>/dev/null; wait $PID 2>/dev/null || true
echo "   -> $(ls -lh "$OUT/bruteforce_ssh_ftp.pcap" | awk '{print $5}')"

# 5 - ZERO-DAY (ICMP flood)
echo "[5/5] Zero-day ICMP flood..."
tcpdump -i lo -w "$OUT/zeroday_icmp_flood.pcap" -c 60 &>/dev/null &
PID=$!; sleep 0.5
timeout 3 hping3 --icmp --faster -c 30 127.0.0.1 2>/dev/null || true
sleep 1; kill $PID 2>/dev/null; wait $PID 2>/dev/null || true
echo "   -> $(ls -lh "$OUT/zeroday_icmp_flood.pcap" | awk '{print $5}')"

echo ""
echo "=== Done. Test PCAPs in $OUT ==="
ls -lh "$OUT/"