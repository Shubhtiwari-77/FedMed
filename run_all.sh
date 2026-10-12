#!/usr/bin/env bash
# Reproduce every number and figure in the report, in order.
set -e
cd "$(dirname "$0")"
echo "[1/5] federated + record-level DP (main result)"
python3 src/run_simulation.py
echo "[2/5] centralized baseline (upper bound)"
python3 src/run_centralized.py
echo "[3/5] ablation: no differential privacy"
python3 src/run_simulation.py --no-dp
echo "[4/5] ablation: client-level DP collapses at K=3"
python3 src/run_simulation.py --dp-mode client --sigma 1.0 --rounds 15
echo "[5/5] privacy-utility sweep"
python3 src/run_simulation.py --privacy-sweep
echo
echo "Done. Everything is in outputs/"
