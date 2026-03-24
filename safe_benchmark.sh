#!/bin/bash
# Safe ExpertFlow Benchmark Runner
# Runs benchmarks ONE AT A TIME with memory guards
export PATH="/opt/homebrew/bin:/usr/local/bin:/Users/jhammant/.asdf/shims:/Users/jhammant/.local/bin:$PATH"
source ~/mlx-env/bin/activate
source ~/dev/expertflow/memory_guard.sh

echo ""
echo "============================================"
echo "  ExpertFlow Safe Benchmark Runner"
echo "  $(date)"
echo "============================================"

# Step 1: Pre-flight
echo ""
echo "--- Step 1: Pre-flight check ---"
preflight_check 20 || { echo "ABORT: Free memory first!"; exit 1; }

# Step 2: Kill any lingering heavy processes
echo ""
echo "--- Step 2: Cleanup ---"
pkill -f "mega_benchmark\|expert_predictor\|snapshot_download" 2>/dev/null
sleep 2
echo "Free after cleanup: $(get_free_gb) GB"

# Step 3: Run benchmarks sequentially
cd ~/dev/expertflow

for MODEL_DIR in ~/models/deepseek-v3.1-4bit ~/models/minimax-m2-4bit ~/models/glm-4.5-4bit; do
    MODEL_NAME=$(basename $MODEL_DIR)
    SHARDS=$(find $MODEL_DIR -name "model*.safetensors" 2>/dev/null | wc -l | tr -d ' ')
    SIZE=$(du -sh $MODEL_DIR 2>/dev/null | awk '{print $1}')
    
    echo ""
    echo "============================================"
    echo "  Benchmarking: $MODEL_NAME ($SIZE, $SHARDS shards)"
    echo "============================================"
    
    if [ "$SHARDS" -lt 5 ]; then
        echo "⏳ Not enough shards — skipping (download incomplete)"
        continue
    fi
    
    # Pre-flight before each model
    preflight_check 15 || { echo "⚠️ Low memory — skipping $MODEL_NAME"; continue; }
    
    echo "Starting benchmark..."
    PYTHONUNBUFFERED=1 python3 -u -c "
import mega_benchmark as mb
import json, time, resource, os

model_map = {
    'deepseek-v3.1-4bit': ('DeepSeek V3.1', '671B', '37B'),
    'minimax-m2-4bit': ('MiniMax-M2', '230B', '10B'),
    'glm-4.5-4bit': ('GLM-4.5 Full', '355B', '32B'),
}

dirname = os.path.basename('$MODEL_DIR')
name, total, active = model_map.get(dirname, (dirname, '?', '?'))

model = {
    'name': name,
    'path': '$MODEL_DIR',
    'total_params': total,
    'active_params': active,
}

# Set soft memory limit (80GB — leaves 48GB for system + apps)
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (80 * 1024**3, -1))
    print(f'Memory limit set: 80GB soft')
except Exception as e:
    print(f'Could not set memory limit: {e}')

t0 = time.time()
result = mb.benchmark_model(model)
elapsed = time.time() - t0

if result:
    outfile = f'/Users/jhammant/dev/expertflow/{dirname}-benchmark.json'
    with open(outfile, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\\n✅ {name} complete in {elapsed:.0f}s → {outfile}')
else:
    print(f'\\n❌ {name} benchmark failed')
" &
    
    BM_PID=$!
    echo "Benchmark PID: $BM_PID"
    
    # Run memory guard in background
    memory_guard_start $BM_PID "$MODEL_NAME" &
    GUARD_PID=$!
    
    # Wait for benchmark to finish
    wait $BM_PID
    BM_EXIT=$?
    kill $GUARD_PID 2>/dev/null
    
    if [ $BM_EXIT -ne 0 ]; then
        echo "⚠️ $MODEL_NAME benchmark exited with code $BM_EXIT"
    fi
    
    # Cool down — let memory settle
    echo "Cooling down (10s)..."
    sleep 10
    echo "Free memory: $(get_free_gb) GB"
done

echo ""
echo "============================================"
echo "  All benchmarks complete!"
echo "  $(date)"
echo "============================================"

# Combine results
python3 -u -c "
import json, glob, os

results = []
for f in sorted(glob.glob(os.path.expanduser('~/dev/expertflow/*-benchmark.json'))):
    if 'mega' not in f:
        with open(f) as fp:
            results.append(json.load(fp))

if results:
    outpath = os.path.expanduser('~/dev/expertflow/mega-benchmark.json')
    with open(outpath, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'Combined {len(results)} results → {outpath}')
"
