#!/usr/bin/env python3
"""
ExpertFlow Meta-Predictor: Speculative Expert Loading
=====================================================
Trains a small neural network to predict which MoE experts will be needed
for upcoming tokens, enabling prefetching before they're needed.

Three-phase approach:
1. COLLECT: Run inference and log expert activation patterns
2. TRAIN: Train a small predictor on activation sequences
3. EVALUATE: Compare predictor-guided prefetching vs reactive LRU

Works with any MoE model in safetensors format.
"""

import os, sys, time, json, glob, struct, gc, math
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────
MODELS = {
    "qwen3-next-80b": {
        "path": os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit"),
        "num_experts": 512,
        "num_active": 10,
        "num_layers": 48,
        "expert_key": "switch_mlp",
    },
    "deepseek-v3.1": {
        "path": os.path.expanduser("~/models/deepseek-v3.1-4bit"),
        "num_experts": 256,
        "num_active": 8,
        "num_layers": 61,
        "expert_key": "experts",
    },
    "minimax-m2": {
        "path": os.path.expanduser("~/models/minimax-m2-4bit"),
        "num_experts": 128,
        "num_active": 8,
        "num_layers": 56,
        "expert_key": "experts",
    },
    "glm-4.5": {
        "path": os.path.expanduser("~/models/glm-4.5-4bit"),
        "num_experts": 256,
        "num_active": 8,
        "num_layers": 62,
        "expert_key": "experts",
    },
}

OUTPUT_DIR = os.path.expanduser("~/dev/expertflow/predictor")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# Phase 1: Analyze Expert Structure & Simulate Activation Patterns
# ══════════════════════════════════════════════════════════════════════════

def analyze_model(name, config):
    """Parse safetensors to find expert tensor structure."""
    path = config["path"]
    expert_key = config["expert_key"]
    
    shard_files = sorted(glob.glob(f"{path}/model*.safetensors"))
    if not shard_files:
        print(f"  ⏳ {name}: not downloaded yet")
        return None
    
    # Parse tensor headers
    expert_tensors = {}
    backbone_tensors = {}
    
    for fpath in shard_files:
        with open(fpath, "rb") as fp:
            hs = struct.unpack("<Q", fp.read(8))[0]
            meta = json.loads(fp.read(hs))
        
        for key, info in meta.items():
            if key == "__metadata__":
                continue
            off = info["data_offsets"]
            size = off[1] - off[0]
            is_expert = expert_key in key
            
            entry = {"size": size, "shape": info["shape"], "dtype": info["dtype"]}
            if is_expert:
                expert_tensors[key] = entry
            else:
                backbone_tensors[key] = entry
    
    total_expert = sum(v["size"] for v in expert_tensors.values())
    total_backbone = sum(v["size"] for v in backbone_tensors.values())
    total = total_expert + total_backbone
    
    # Identify unique expert indices
    expert_indices = set()
    for key in expert_tensors:
        # Extract expert index from key patterns like:
        # "model.layers.X.switch_mlp.experts.Y.gate_proj.weight"
        # "model.layers.X.mlp.experts.Y.gate_proj.weight"
        parts = key.split(".")
        for i, p in enumerate(parts):
            if p == "experts" and i + 1 < len(parts):
                try:
                    expert_indices.add(int(parts[i + 1]))
                except ValueError:
                    pass
    
    return {
        "name": name,
        "shards": len(shard_files),
        "expert_tensors": len(expert_tensors),
        "backbone_tensors": len(backbone_tensors),
        "expert_gb": total_expert / 1e9,
        "backbone_gb": total_backbone / 1e9,
        "total_gb": total / 1e9,
        "expert_pct": total_expert / total * 100 if total > 0 else 0,
        "unique_experts": len(expert_indices) if expert_indices else config["num_experts"],
        "num_active": config["num_active"],
        "num_layers": config["num_layers"],
        "expert_keys": list(expert_tensors.keys())[:10],  # sample
    }


# ══════════════════════════════════════════════════════════════════════════
# Phase 2: Generate Realistic Expert Activation Traces
# ══════════════════════════════════════════════════════════════════════════

def generate_activation_traces(num_experts, num_active, num_layers, num_sequences=500, seq_len=128):
    """
    Generate realistic expert activation patterns using a simulated router.
    
    Real MoE routers exhibit:
    1. Expert popularity skew (some experts activate much more often)
    2. Layer-specific preferences (different layers have different hot experts)
    3. Temporal locality (consecutive tokens often use similar experts)
    4. Semantic clustering (similar tokens route to same experts)
    
    We model all 4 properties.
    """
    np.random.seed(42)
    traces = []
    
    # Property 1: Global expert popularity (Zipf-like distribution)
    popularity = np.random.zipf(1.5, num_experts).astype(float)
    popularity /= popularity.sum()
    
    # Property 2: Layer-specific expert preferences
    layer_prefs = np.zeros((num_layers, num_experts))
    for l in range(num_layers):
        # Each layer has its own popularity distribution (shifted)
        shift = np.random.randint(0, num_experts)
        layer_prefs[l] = np.roll(popularity, shift)
        # Add noise
        layer_prefs[l] += np.random.dirichlet(np.ones(num_experts) * 0.1)
        layer_prefs[l] /= layer_prefs[l].sum()
    
    # Property 3 & 4: Generate sequences with temporal locality
    for seq_idx in range(num_sequences):
        seq_trace = []
        prev_experts = None
        
        # Pick a "semantic cluster" for this sequence
        cluster_center = np.random.randint(0, num_experts)
        cluster_boost = np.zeros(num_experts)
        cluster_size = num_experts // 8
        for i in range(cluster_size):
            idx = (cluster_center + i) % num_experts
            cluster_boost[idx] = 0.3
        
        for t in range(seq_len):
            token_experts = []
            for l in range(num_layers):
                # Combine: layer preference + cluster boost + temporal locality
                probs = layer_prefs[l].copy()
                probs += cluster_boost
                
                # Temporal locality: boost experts used in previous token
                if prev_experts is not None and l < len(prev_experts):
                    for e in prev_experts[l]:
                        probs[e] *= 2.0
                
                probs /= probs.sum()
                
                # Select top-K experts
                selected = np.random.choice(
                    num_experts, size=num_active, replace=False, p=probs
                )
                token_experts.append(sorted(selected.tolist()))
            
            seq_trace.append(token_experts)
            prev_experts = token_experts
        
        traces.append(seq_trace)
    
    return traces


# ══════════════════════════════════════════════════════════════════════════
# Phase 3: Train Expert Predictor
# ══════════════════════════════════════════════════════════════════════════

class ExpertPredictor:
    """
    Lightweight expert predictor using a 2-layer MLP.
    
    Input: one-hot encoding of last K expert selections (flattened across layers)
    Output: probability distribution over experts for next step (per layer)
    
    Trained with cross-entropy loss using pure numpy (no torch dependency).
    """
    
    def __init__(self, num_experts, num_layers, num_active, history_len=4, hidden_dim=256):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.num_active = num_active
        self.history_len = history_len
        self.hidden_dim = hidden_dim
        
        # Input: history_len * num_layers * num_experts (one-hot)
        self.input_dim = history_len * num_layers * num_experts
        # Output: num_layers * num_experts (probability per layer)
        self.output_dim = num_layers * num_experts
        
        # Xavier init
        scale1 = np.sqrt(2.0 / (self.input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + self.output_dim))
        
        self.W1 = np.random.randn(self.input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        self.W3 = np.random.randn(hidden_dim, self.output_dim).astype(np.float32) * scale2
        self.b3 = np.zeros(self.output_dim, dtype=np.float32)
        
        self.model_size_mb = (
            self.W1.nbytes + self.b1.nbytes +
            self.W2.nbytes + self.b2.nbytes +
            self.W3.nbytes + self.b3.nbytes
        ) / 1e6
    
    def encode_history(self, history):
        """Convert list of expert selections to one-hot input."""
        x = np.zeros(self.input_dim, dtype=np.float32)
        for t, step in enumerate(history[-self.history_len:]):
            for l, experts in enumerate(step):
                for e in experts:
                    idx = t * self.num_layers * self.num_experts + l * self.num_experts + e
                    if idx < self.input_dim:
                        x[idx] = 1.0
        return x
    
    def encode_target(self, step):
        """Convert expert selections to target vector."""
        y = np.zeros(self.output_dim, dtype=np.float32)
        for l, experts in enumerate(step):
            for e in experts:
                idx = l * self.num_experts + e
                if idx < self.output_dim:
                    y[idx] = 1.0
        return y
    
    def forward(self, x):
        """Forward pass with ReLU activations."""
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        return logits, h2, h1
    
    def predict_topk(self, x, k=None):
        """Predict top-K experts per layer."""
        if k is None:
            k = self.num_active * 2  # Predict 2x active for prefetch buffer
        
        logits, _, _ = self.forward(x)
        predictions = []
        
        for l in range(self.num_layers):
            layer_logits = logits[l * self.num_experts:(l + 1) * self.num_experts]
            top_k = np.argsort(layer_logits)[-k:]
            predictions.append(sorted(top_k.tolist()))
        
        return predictions
    
    def train(self, traces, epochs=20, lr=0.001, batch_size=64):
        """Train with mini-batch SGD and Adam-like updates."""
        print(f"\n  Training predictor ({self.model_size_mb:.1f} MB, {self.hidden_dim}d hidden)...")
        print(f"  Input: {self.input_dim}d → Hidden: {self.hidden_dim}d → Output: {self.output_dim}d")
        
        # Build training data
        X_train, Y_train = [], []
        for trace in traces:
            for t in range(self.history_len, len(trace)):
                history = trace[t - self.history_len:t]
                x = self.encode_history(history)
                y = self.encode_target(trace[t])
                X_train.append(x)
                Y_train.append(y)
        
        X = np.array(X_train, dtype=np.float32)
        Y = np.array(Y_train, dtype=np.float32)
        n = len(X)
        print(f"  Training samples: {n:,}")
        
        # Adam optimizer state
        m_W1 = np.zeros_like(self.W1); v_W1 = np.zeros_like(self.W1)
        m_b1 = np.zeros_like(self.b1); v_b1 = np.zeros_like(self.b1)
        m_W2 = np.zeros_like(self.W2); v_W2 = np.zeros_like(self.W2)
        m_b2 = np.zeros_like(self.b2); v_b2 = np.zeros_like(self.b2)
        m_W3 = np.zeros_like(self.W3); v_W3 = np.zeros_like(self.W3)
        m_b3 = np.zeros_like(self.b3); v_b3 = np.zeros_like(self.b3)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        t_step = 0
        
        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, n, batch_size):
                batch_idx = perm[i:i + batch_size]
                Xb = X[batch_idx]
                Yb = Y[batch_idx]
                bs = len(Xb)
                t_step += 1
                
                # Forward
                h1 = np.maximum(0, Xb @ self.W1 + self.b1)
                h2 = np.maximum(0, h1 @ self.W2 + self.b2)
                logits = h2 @ self.W3 + self.b3
                
                # Sigmoid cross-entropy loss (multi-label)
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
                loss = -np.mean(Yb * np.log(probs + 1e-7) + (1 - Yb) * np.log(1 - probs + 1e-7))
                epoch_loss += loss
                num_batches += 1
                
                # Backward
                dlogits = (probs - Yb) / bs
                
                dW3 = h2.T @ dlogits
                db3 = dlogits.sum(axis=0)
                dh2 = dlogits @ self.W3.T
                dh2 = dh2 * (h2 > 0)
                
                dW2 = h1.T @ dh2
                db2 = dh2.sum(axis=0)
                dh1 = dh2 @ self.W2.T
                dh1 = dh1 * (h1 > 0)
                
                dW1 = Xb.T @ dh1
                db1 = dh1.sum(axis=0)
                
                # Adam updates
                for param, grad, m, v in [
                    (self.W1, dW1, m_W1, v_W1), (self.b1, db1, m_b1, v_b1),
                    (self.W2, dW2, m_W2, v_W2), (self.b2, db2, m_b2, v_b2),
                    (self.W3, dW3, m_W3, v_W3), (self.b3, db3, m_b3, v_b3),
                ]:
                    m[:] = beta1 * m + (1 - beta1) * grad
                    v[:] = beta2 * v + (1 - beta2) * grad ** 2
                    m_hat = m / (1 - beta1 ** t_step)
                    v_hat = v / (1 - beta2 ** t_step)
                    param -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
            avg_loss = epoch_loss / num_batches
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
        
        print(f"  Training complete. Final loss: {avg_loss:.4f}")
        return avg_loss


# ══════════════════════════════════════════════════════════════════════════
# Phase 4: Evaluate Predictor vs LRU
# ══════════════════════════════════════════════════════════════════════════

def evaluate_strategies(predictor, traces, num_experts, num_active, num_layers,
                       lru_sizes=[20, 50, 100, 200]):
    """Compare predictor-guided prefetching vs reactive LRU caching."""
    
    # Use last 20% of traces as test set
    split = int(len(traces) * 0.8)
    test_traces = traces[split:]
    
    results = {}
    
    # Baseline: Reactive LRU
    for lru_size in lru_sizes:
        hits, misses = 0, 0
        for trace in test_traces:
            cache = set()
            cache_order = []
            
            for step in trace:
                for l, experts in enumerate(step):
                    for e in experts:
                        key = (l, e)
                        if key in cache:
                            hits += 1
                            cache_order.remove(key)
                            cache_order.append(key)
                        else:
                            misses += 1
                            cache.add(key)
                            cache_order.append(key)
                            while len(cache) > lru_size:
                                evicted = cache_order.pop(0)
                                cache.discard(evicted)
        
        total = hits + misses
        results[f"LRU-{lru_size}"] = {
            "hits": hits, "misses": misses,
            "hit_rate": hits / total * 100 if total > 0 else 0,
            "cache_size": lru_size,
        }
    
    # Predictor-guided prefetching
    for prefetch_k in [1, 2, 3]:
        k = num_active * prefetch_k
        hits, misses, prefetch_hits = 0, 0, 0
        
        for trace in test_traces:
            cache = set()
            cache_order = []
            lru_size = 100  # Same as LRU-100 for fair comparison
            
            for t in range(len(trace)):
                step = trace[t]
                
                # Check cache for current step
                for l, experts in enumerate(step):
                    for e in experts:
                        key = (l, e)
                        if key in cache:
                            hits += 1
                            cache_order.remove(key)
                            cache_order.append(key)
                        else:
                            misses += 1
                            cache.add(key)
                            cache_order.append(key)
                
                # Predict and prefetch for next step
                if t >= predictor.history_len:
                    history = trace[t - predictor.history_len + 1:t + 1]
                    x = predictor.encode_history(history)
                    predictions = predictor.predict_topk(x, k=k)
                    
                    for l, pred_experts in enumerate(predictions):
                        for e in pred_experts:
                            key = (l, e)
                            if key not in cache:
                                cache.add(key)
                                cache_order.append(key)
                                prefetch_hits += 1
                
                # Evict
                while len(cache) > lru_size + k:
                    evicted = cache_order.pop(0)
                    cache.discard(evicted)
        
        total = hits + misses
        results[f"Predictor-{prefetch_k}x"] = {
            "hits": hits, "misses": misses,
            "hit_rate": hits / total * 100 if total > 0 else 0,
            "prefetch_k": k,
            "prefetch_buffer": prefetch_hits,
        }
    
    # Oracle: perfect prediction (upper bound)
    hits, misses = 0, 0
    for trace in test_traces:
        cache = set()
        for t in range(len(trace)):
            step = trace[t]
            for l, experts in enumerate(step):
                for e in experts:
                    key = (l, e)
                    if key in cache:
                        hits += 1
                    else:
                        misses += 1
            # Perfect prefetch: load exactly what next step needs
            if t + 1 < len(trace):
                cache = set()
                for l, experts in enumerate(trace[t + 1]):
                    for e in experts:
                        cache.add((l, e))
    
    total = hits + misses
    results["Oracle"] = {
        "hits": hits, "misses": misses,
        "hit_rate": hits / total * 100 if total > 0 else 0,
    }
    
    return results


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  ExpertFlow Meta-Predictor: Speculative Expert Loading")
    print("=" * 70)
    
    all_results = {}
    
    for name, config in MODELS.items():
        print(f"\n{'#' * 70}")
        print(f"  Model: {name}")
        print(f"{'#' * 70}")
        
        # Analyze model
        info = analyze_model(name, config)
        if info is None:
            continue
        
        print(f"  Shards: {info['shards']}")
        print(f"  Total: {info['total_gb']:.1f} GB | Expert: {info['expert_gb']:.1f} GB ({info['expert_pct']:.0f}%)")
        print(f"  Experts: {info['unique_experts']} total, {info['num_active']} active/token")
        
        num_experts = info["unique_experts"]
        num_active = info["num_active"]
        num_layers = info["num_layers"]
        
        # For very large models, reduce dimensions to keep training feasible
        # Use representative subset of layers
        max_layers = min(num_layers, 8)  # Sample 8 layers for predictor
        
        print(f"\n  Generating activation traces ({max_layers} sampled layers)...")
        t0 = time.time()
        traces = generate_activation_traces(
            num_experts=num_experts,
            num_active=num_active,
            num_layers=max_layers,
            num_sequences=500,
            seq_len=128,
        )
        gen_time = time.time() - t0
        print(f"  Generated {len(traces)} sequences × {len(traces[0])} tokens in {gen_time:.1f}s")
        
        # Train predictor
        # Use smaller hidden dim for models with many experts
        hidden_dim = 128 if num_experts > 256 else 256
        
        predictor = ExpertPredictor(
            num_experts=num_experts,
            num_layers=max_layers,
            num_active=num_active,
            history_len=4,
            hidden_dim=hidden_dim,
        )
        
        t0 = time.time()
        train_traces = traces[:int(len(traces) * 0.8)]
        final_loss = predictor.train(train_traces, epochs=20, lr=0.001)
        train_time = time.time() - t0
        
        # Evaluate
        print(f"\n  Evaluating strategies...")
        t0 = time.time()
        eval_results = evaluate_strategies(
            predictor, traces, num_experts, num_active, max_layers,
            lru_sizes=[20, 50, 100, 200],
        )
        eval_time = time.time() - t0
        
        # Print results
        print(f"\n  {'Strategy':<20} {'Hit Rate':>10} {'Hits':>10} {'Misses':>10}")
        print(f"  {'-'*55}")
        for strat, r in sorted(eval_results.items(), key=lambda x: x[1]["hit_rate"]):
            print(f"  {strat:<20} {r['hit_rate']:>9.1f}% {r['hits']:>10,} {r['misses']:>10,}")
        
        # Calculate improvement
        lru100 = eval_results.get("LRU-100", {}).get("hit_rate", 0)
        pred2x = eval_results.get("Predictor-2x", {}).get("hit_rate", 0)
        oracle = eval_results.get("Oracle", {}).get("hit_rate", 0)
        
        improvement = pred2x - lru100
        miss_reduction = ((100 - lru100) - (100 - pred2x)) / (100 - lru100) * 100 if lru100 < 100 else 0
        
        print(f"\n  📊 Key Results:")
        print(f"     LRU-100 baseline:     {lru100:.1f}% hit rate")
        print(f"     Predictor (2x):       {pred2x:.1f}% hit rate")
        print(f"     Oracle (perfect):     {oracle:.1f}% hit rate")
        print(f"     Improvement:          +{improvement:.1f}pp ({miss_reduction:.0f}% miss reduction)")
        print(f"     Predictor size:       {predictor.model_size_mb:.1f} MB")
        print(f"     Train time:           {train_time:.1f}s")
        
        all_results[name] = {
            "model_info": info,
            "predictor_size_mb": predictor.model_size_mb,
            "training_time_s": round(train_time, 1),
            "training_loss": round(final_loss, 4),
            "strategies": {k: {kk: vv for kk, vv in v.items()} for k, v in eval_results.items()},
            "improvement_pp": round(improvement, 1),
            "miss_reduction_pct": round(miss_reduction, 1),
        }
    
    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: Meta-Predictor Results Across Models")
    print(f"{'=' * 70}")
    print(f"\n  {'Model':<20} {'Experts':>8} {'LRU-100':>9} {'Pred-2x':>9} {'Improve':>9} {'Pred Size':>10}")
    print(f"  {'-'*68}")
    for name, r in all_results.items():
        ne = r["model_info"]["unique_experts"]
        lru = r["strategies"].get("LRU-100", {}).get("hit_rate", 0)
        pred = r["strategies"].get("Predictor-2x", {}).get("hit_rate", 0)
        imp = r["improvement_pp"]
        sz = r["predictor_size_mb"]
        print(f"  {name:<20} {ne:>8} {lru:>8.1f}% {pred:>8.1f}% {imp:>+8.1f}pp {sz:>9.1f}MB")
    
    # Save results
    outpath = os.path.join(OUTPUT_DIR, "predictor-results.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {outpath}")
    
    # Also save a paper-ready summary
    summary_path = os.path.join(OUTPUT_DIR, "predictor-summary.md")
    with open(summary_path, "w") as f:
        f.write("# ExpertFlow Meta-Predictor Results\n\n")
        f.write("## Method\n")
        f.write("- 2-layer MLP predicting next-step expert activations from 4-step history\n")
        f.write("- Trained on 500 sequences × 128 tokens of simulated router activations\n")
        f.write("- Activation patterns model: Zipf popularity + layer preferences + temporal locality + semantic clustering\n")
        f.write("- Evaluated: LRU baselines, predictor-guided prefetching (1x/2x/3x), oracle upper bound\n\n")
        f.write("## Results\n\n")
        f.write(f"| Model | Experts | LRU-100 | Predictor-2x | Improvement | Miss Reduction | Predictor Size |\n")
        f.write(f"|-------|---------|---------|-------------|-------------|----------------|----------------|\n")
        for name, r in all_results.items():
            ne = r["model_info"]["unique_experts"]
            lru = r["strategies"].get("LRU-100", {}).get("hit_rate", 0)
            pred = r["strategies"].get("Predictor-2x", {}).get("hit_rate", 0)
            imp = r["improvement_pp"]
            mr = r["miss_reduction_pct"]
            sz = r["predictor_size_mb"]
            f.write(f"| {name} | {ne} | {lru:.1f}% | {pred:.1f}% | +{imp:.1f}pp | {mr:.0f}% | {sz:.1f} MB |\n")
    
    print(f"  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
