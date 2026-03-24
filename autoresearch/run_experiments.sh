#!/bin/bash
# ExpertFlow AutoResearch Runner
# Runs benchmark, saves results, tracks progress

set -e
export PATH="/opt/homebrew/bin:/usr/local/bin:/Users/jhammant/.asdf/shims:/Users/jhammant/.local/bin:$PATH"
source ~/mlx-env/bin/activate
cd ~/dev/expertflow/autoresearch

MODEL=${1:-glm}
LABEL=${2:-baseline}

echo "=== ExpertFlow Experiment: $LABEL ($MODEL) ==="
echo "Time: $(date)"

# Run benchmark
python3 -u benchmark.py --model $MODEL 2>&1 | tee /tmp/ef_bench_${LABEL}.log

# Append to experiment log
RESULT_FILE=$(ls -t ~/dev/expertflow/bench_*.json 2>/dev/null | head -1)
if [ -f "$RESULT_FILE" ]; then
    # Add label to result
    python3 -c "
import json
with open('$RESULT_FILE') as f: d = json.load(f)
d['label'] = '$LABEL'
print(json.dumps(d))
" >> ~/dev/expertflow/autoresearch/experiments.jsonl
    echo "Logged to experiments.jsonl"
fi
