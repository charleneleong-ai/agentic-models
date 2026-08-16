#!/usr/bin/env bash
# Queue the Dyck depth sweep — waits for the current GPU job to finish, then launches.
#
# Usage: ssh pi-a100-80gb 'bash -s' < scripts/queue_dyck_sweep.sh
# Or:    bash scripts/queue_dyck_sweep.sh   (on the remote directly)
#
# The current occupant is PID 1076887 (small-smart-models routing×quant diagnostic).
# The script polls nvidia-smi for GPU memory to drop below 5GB, then starts the sweep.

set -euo pipefail

REPO=~/agentic-models
LOG_DIR=~/agentic-models/logs
WAIT_PID=${1:-1076887}             # PID of the job to wait for
GPU_UTIL_THRESHOLD=5000            # MiB — launch when VRAM drops below this
POLL_INTERVAL=60                   # seconds between checks
SWEEP_NAME=attn-res-dyck

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="$LOG_DIR/dyck_sweep_${TIMESTAMP}.log"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_FILE"; }

log "=== Dyck depth sweep queue ==="
log "Waiting for PID $WAIT_PID to release the GPU..."

# --- Wait for the current job ---
# Wait for GPU memory to drop below threshold, regardless of which PID holds it.
# The target PID may exit but a child or restart may grab the GPU immediately after.
while true; do
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)

    # Check if GPU is free
    if [ -n "$GPU_MEM" ] && [ "$GPU_MEM" -lt "$GPU_UTIL_THRESHOLD" ]; then
        log "GPU memory dropped to ${GPU_MEM} MiB (threshold: ${GPU_UTIL_THRESHOLD})."
        break
    fi

    # Also check if the target PID is gone AND GPU memory is declining
    if ! kill -0 "$WAIT_PID" 2>/dev/null; then
        log "PID $WAIT_PID has exited. GPU still at ${GPU_MEM:-?} MiB, waiting for cleanup..."
    fi

    log "  GPU: ${GPU_MEM:-?} MiB, util: ${GPU_UTIL:-?}%. Waiting ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
done

# Give a short grace period for process cleanup and GPU memory release
sleep 15

# --- Pre-flight ---
log "=== Pre-flight ==="
cd "$REPO"
git pull --ff-only
log "Code synced: $(git rev-parse --short HEAD)"

# Verify the sweep config
log "Sweep config:"
grep -E "depth:|seq_len:|steps:" configs/ablations/attn-res-dyck.yaml | tee -a "$LOG_FILE" || true

# Check GPU is actually free
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv | tee -a "$LOG_FILE"

# --- Run the sweep ---
log "=== Launching Dyck depth sweep ==="
log "Config: nesting 64, 2400 steps, 4 depths × 3 arms × 2 seeds = 24 runs"
log "Estimated time: ~18 GPU-hours"
log "Log: $LOG_FILE"

# Run with deterministic algorithms ON (repo convention)
cd "$REPO"
log "Starting sweep at $(date -u +%Y-%m-%dT%H:%M:%SZ)..."
PATH="$REPO/.venv/bin:$PATH" CUBLAS_WORKSPACE_CONFIG=:4096:8 archlab ablate attn-res-dyck --device cuda 2>&1 | tee -a "$LOG_FILE"
SWEEP_EXIT=${PIPESTATUS[0]}

log "=== Sweep finished with exit code $SWEEP_EXIT ==="
log "Finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [ "$SWEEP_EXIT" -eq 0 ]; then
    log "Results:"
    ls -la experiments/attn-res-dyck/ 2>/dev/null | tee -a "$LOG_FILE"
    # Print a summary of results
    if [ -f experiments/attn-res-dyck/results.jsonl ]; then
        log "Result rows:"
        wc -l experiments/attn-res-dyck/results.jsonl | tee -a "$LOG_FILE"
    fi
else
    log "Sweep failed. Check the log above for errors."
fi

log "=== Queue script complete ==="
