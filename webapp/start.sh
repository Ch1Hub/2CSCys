#!/bin/bash
# 2CSCys Web App — Single-process deployment
# Build frontend first, then serve everything via Flask on port 5000
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Build frontend
echo "[build] Compiling React frontend..."
cd webapp/frontend && npm install --silent && npx vite build
cd "$ROOT"

echo ""
echo "=== 2CSCys NIDS Dashboard ==="
echo ""
echo "Open in browser:  http://$(hostname -I | awk '{print $1}'):5000"
echo ""

venv/bin/python webapp/backend/app.py
