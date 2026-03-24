#!/bin/bash
# ============================================================================
# GPU Job Scheduler for Ablation Study
# ============================================================================
# Runs all 64 configs (8 aggregators × 8 losses) across 4 GPUs.
# Automatically starts new jobs as GPUs become free.
#
# Usage:
#   cd /path/to/liver-pdff-mil
#   ./run_ablation.sh
#
# Configs: 64 experiments
#   Aggregators: mean, max, attention, gated, abmil, mh_gated4, mh_gated8, temporal_conv32h4
#   Losses: l1, l2, huber, logcosh, wing, focal, weighted_zone, threshold_aware
#   All with: use_classifier=true, cls_weight=0.3, n_folds=5
# ============================================================================

set -e

# Configuration
CONFIG_DIR="configs/ablation"
LOG_DIR="logs/ablation"
GPUS=(0 1 2 3)
POLL_INTERVAL=60  # Check every 60 seconds

# Create directories
mkdir -p "$LOG_DIR"

# Check if configs exist
if [ ! -f "$CONFIG_DIR/all_configs.txt" ]; then
    echo "ERROR: Config list not found at $CONFIG_DIR/all_configs.txt"
    echo "Run from the liver-pdff-mil directory"
    exit 1
fi

# Read configs into array
mapfile -t CONFIGS < "$CONFIG_DIR/all_configs.txt"
TOTAL=${#CONFIGS[@]}

echo "=============================================="
echo "ABLATION STUDY"
echo "=============================================="
echo "Total experiments: $TOTAL"
echo "GPUs: ${GPUS[*]}"
echo "Config directory: $CONFIG_DIR"
echo "Log directory: $LOG_DIR"
echo "=============================================="
echo ""

# Track GPU assignments
declare -A gpu_pid
declare -A gpu_config
declare -A gpu_start

for gpu in "${GPUS[@]}"; do
    gpu_pid[$gpu]=""
    gpu_config[$gpu]=""
    gpu_start[$gpu]=""
done

# Counters
NEXT_JOB=0
COMPLETED=0
FAILED=0

# Timestamp function
ts() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Check if process is running
is_running() {
    kill -0 "$1" 2>/dev/null
}

# Start a job on a GPU
start_job() {
    local gpu=$1
    local config_name=$2
    local config_path="$CONFIG_DIR/${config_name}.yaml"
    local log_file="$LOG_DIR/${config_name}.log"

    echo "[$(ts)] GPU $gpu: Starting $config_name"

    CUDA_VISIBLE_DEVICES=$gpu python src/main.py "$config_path" > "$log_file" 2>&1 &

    gpu_pid[$gpu]=$!
    gpu_config[$gpu]=$config_name
    gpu_start[$gpu]=$(date +%s)
}

# Check GPUs and schedule jobs
check_and_schedule() {
    for gpu in "${GPUS[@]}"; do
        local pid=${gpu_pid[$gpu]}

        if [ -z "$pid" ] || ! is_running "$pid"; then
            # Previous job finished
            if [ -n "$pid" ]; then
                wait "$pid" 2>/dev/null
                local exit_code=$?
                local config_name=${gpu_config[$gpu]}
                local start_time=${gpu_start[$gpu]}
                local end_time=$(date +%s)
                local duration=$((end_time - start_time))
                local duration_min=$((duration / 60))

                if [ $exit_code -eq 0 ]; then
                    echo "[$(ts)] GPU $gpu: ✓ Completed $config_name (${duration_min}m)"
                    ((COMPLETED++))
                else
                    echo "[$(ts)] GPU $gpu: ✗ FAILED $config_name (exit $exit_code)"
                    ((FAILED++))
                fi
            fi

            # Start next job
            if [ $NEXT_JOB -lt $TOTAL ]; then
                local next_config=${CONFIGS[$NEXT_JOB]}
                start_job "$gpu" "$next_config"
                ((NEXT_JOB++))
            else
                gpu_pid[$gpu]=""
                gpu_config[$gpu]=""
            fi
        fi
    done
}

# Show status
show_status() {
    local running=0
    local status=""

    for gpu in "${GPUS[@]}"; do
        if [ -n "${gpu_pid[$gpu]}" ] && is_running "${gpu_pid[$gpu]}"; then
            ((running++))
            status+="GPU$gpu:${gpu_config[$gpu]} "
        fi
    done

    local pending=$((TOTAL - NEXT_JOB))
    echo "[$(ts)] Done:$COMPLETED Failed:$FAILED Running:$running Pending:$pending | $status"
}

# Main loop
echo "Starting jobs..."
echo ""

check_and_schedule

while true; do
    sleep $POLL_INTERVAL
    check_and_schedule
    show_status

    # Check if all done
    local all_done=true
    for gpu in "${GPUS[@]}"; do
        if [ -n "${gpu_pid[$gpu]}" ] && is_running "${gpu_pid[$gpu]}"; then
            all_done=false
            break
        fi
    done

    if $all_done && [ $NEXT_JOB -ge $TOTAL ]; then
        break
    fi
done

echo ""
echo "=============================================="
echo "ABLATION COMPLETE"
echo "=============================================="
echo "Total:     $TOTAL"
echo "Completed: $COMPLETED"
echo "Failed:    $FAILED"
echo ""
echo "Results: $LOG_DIR/"
echo "=============================================="

# Summary of failed jobs
if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed jobs (check logs):"
    for log in $LOG_DIR/*.log; do
        name=$(basename "$log" .log)
        if grep -q "Error\|Exception\|Traceback" "$log" 2>/dev/null; then
            echo "  - $name"
        fi
    done
fi
