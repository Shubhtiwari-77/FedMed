#!/usr/bin/env bash
# Launches the server + all 3 hospital clients on ONE machine, in the
# background, so you can watch the real gRPC federation without a second
# laptop. Logs land in outputs/logs/.
#
#   bash federated/run_local_demo.sh 20      # 20 rounds
#
# For the REAL two-laptop demo, do it manually instead — see README Section 6.
set -e
cd "$(dirname "$0")/.."
ROUNDS="${1:-20}"
mkdir -p outputs/logs

echo "Starting server (${ROUNDS} rounds)..."
python3 federated/server.py --rounds "$ROUNDS" --min-clients 3 \
    > outputs/logs/server.log 2>&1 &
SERVER_PID=$!

# The server generates its test set and initialises the model before it starts
# listening, so give it a real head start. Clients that dial too early will get
# "connection refused" and exit.
echo "Waiting 25s for the server to bind..."
sleep 25

for CID in H1_Max_Delhi H2_Apollo_Chennai H3_SKIMS_Srinagar; do
    echo "  starting client $CID"
    python3 federated/client.py --client-id "$CID" --server 127.0.0.1:8080 \
        > "outputs/logs/${CID}.log" 2>&1 &
    sleep 2
done

echo
echo "All nodes launched. Following the server log (Ctrl-C to stop watching):"
echo "-------------------------------------------------------------------"
tail -f outputs/logs/server.log &
TAIL_PID=$!
wait $SERVER_PID
kill $TAIL_PID 2>/dev/null || true
echo
echo "Done. Client logs: outputs/logs/H*.log"
