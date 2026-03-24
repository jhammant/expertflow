#!/bin/bash
# ExpertFlow Memory Guard — prevents OOM by monitoring free memory
# Usage: source memory_guard.sh && preflight_check 20 && memory_guard_start <PID> "name"

THRESHOLD_GB=${MG_THRESHOLD:-16}
CHECK_INTERVAL=5

get_free_gb() {
    vm_stat | awk -v ps=16384 '
    /Pages free/ {free=$NF+0}
    /Pages inactive/ {inactive=$NF+0}
    END {printf "%.1f", (free + inactive) * ps / 1073741824}
    '
}

memory_guard_start() {
    local pid=$1
    local name=${2:-"process"}
    echo "🛡️ Memory guard: PID $pid ($name), kill below ${THRESHOLD_GB}GB free"
    
    while kill -0 $pid 2>/dev/null; do
        FREE=$(get_free_gb)
        FREE_INT=${FREE%.*}
        
        if [ "$FREE_INT" -lt "$THRESHOLD_GB" ]; then
            echo "⚠️ LOW MEMORY: ${FREE}GB free — killing $name (PID $pid)"
            kill -9 $pid 2>/dev/null
            echo "✅ Killed to prevent system freeze"
            return 1
        fi
        sleep $CHECK_INTERVAL
    done
    echo "✅ $name finished. Memory OK."
    return 0
}

preflight_check() {
    local needed_gb=${1:-20}
    local free=$(get_free_gb)
    local free_int=${free%.*}
    
    echo "🔍 Pre-flight: ${free}GB free, need ${needed_gb}GB"
    if [ "$free_int" -lt "$needed_gb" ]; then
        echo "❌ ABORT: Not enough memory. Close LM Studio/Chrome first."
        return 1
    fi
    echo "✅ Memory OK"
    return 0
}

echo "Memory guard loaded. Free: $(get_free_gb) GB"
