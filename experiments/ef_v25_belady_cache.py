#!/usr/bin/env python3
"""
ExpertFlow v25 — Belady-Approximate ML Cache for MoE Expert Offloading
=======================================================================
Inspired by FlashMoE (arXiv 2601.17063): replaces LRU eviction with a
small neural network trained to approximate Belady's optimal replacement.

Key insight: LRU is suboptimal for MoE expert access patterns because
expert routing has structure (some experts are "bursty" — inactive for
many tokens then heavily used). A learned policy can predict which
cached experts won't be needed soon.

Architecture:
  - 3-layer FFN (128 hidden, SiLU) per MoE layer
  - Input: [1/recency, frequency/max_freq] per cached expert
  - Output: eviction score (higher = more likely to evict)
  - Trained on Belady-optimal labels from routing traces

Phases:
  1. Trace collection: run model with LRU cache, log routing decisions
  2. Belady labeling: compute optimal eviction from traces (requires future knowledge)
  3. Train predictor: small FFN learns to approximate Belady from features
  4. Deploy: replace LRU eviction with learned policy

v24 baseline: LRU cache, 300 budget, ~36% hit rate
v25 target: Belady-approximate cache, same budget, ~50%+ hit rate
"""

import os, sys, time, json, math, traceback
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional
import subprocess


def free_gb():
    try:
        out = subprocess.check_output(["vm_stat"], timeout=2).decode()
        f = i = 0
        for l in out.split("\n"):
            if "Pages free" in l: f = int(l.split()[-1].rstrip("."))
            elif "Pages inactive" in l: i = int(l.split()[-1].rstrip("."))
        return (f + i) * 16384 / 1e9
    except:
        return -1


# ─── Belady-Approximate Eviction Policy ───────────────────────────

@dataclass
class ExpertAccessRecord:
    """Track access patterns for one expert across tokens."""
    last_access: int = 0      # Token index of last access
    access_count: int = 0     # Total accesses
    recent_gap: float = 0.0   # Average gap between recent accesses


class BeladyPredictor:
    """Small FFN that predicts eviction scores (approximating Belady's OPT).

    Per FlashMoE paper: 3-layer FFN, 128 hidden, SiLU activation.
    Input: [1/recency, frequency/max_freq] → eviction_score
    """

    def __init__(self, hidden_dim=128):
        self.hidden_dim = hidden_dim
        # Initialize weights (will be overwritten if trained model exists)
        self.w1 = np.random.randn(2, hidden_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        self.w3 = np.random.randn(hidden_dim, 1).astype(np.float32) * 0.1
        self.b3 = np.zeros(1, dtype=np.float32)
        self.trained = False

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict eviction scores. Higher = should evict first.

        features: (N, 2) array of [1/recency, frequency/max_freq]
        returns: (N,) eviction scores
        """
        # Layer 1
        h = features @ self.w1 + self.b1
        h = np.maximum(h, 0) * (1 / (1 + np.exp(-h)))  # SiLU approximation
        # Layer 2
        h = h @ self.w2 + self.b2
        h = np.maximum(h, 0) * (1 / (1 + np.exp(-h)))
        # Layer 3
        scores = (h @ self.w3 + self.b3).squeeze(-1)
        return scores

    def save(self, path):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                 w3=self.w3, b3=self.b3, trained=np.array([self.trained]))

    def load(self, path):
        data = np.load(path)
        self.w1, self.b1 = data['w1'], data['b1']
        self.w2, self.b2 = data['w2'], data['b2']
        self.w3, self.b3 = data['w3'], data['b3']
        self.trained = bool(data['trained'][0])


class BeladyExpertCache:
    """Expert cache with Belady-approximate eviction policy."""

    def __init__(self, budget=300, use_learned=True):
        self.cache = OrderedDict()
        self.budget = budget
        self.use_learned = use_learned

        # Access tracking
        self.access_records: dict[tuple, ExpertAccessRecord] = {}
        self.current_token = 0
        self.max_freq = 1

        # Routing trace for offline training
        self.routing_trace: list[list[tuple[int, int]]] = []  # per-token list of (layer, expert)

        # Predictor
        self.predictor = BeladyPredictor()

        # Stats
        self.token_hits = 0
        self.token_misses = 0
        self.total_hits = 0
        self.total_misses = 0
        self.eviction_count = 0

    def get(self, key):
        if key in self.cache:
            self.token_hits += 1
            self.total_hits += 1
            self._update_access(key)
            return self.cache[key]
        self.token_misses += 1
        self.total_misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self._update_access(key)
        else:
            self.cache[key] = value
            self._update_access(key)

    def _update_access(self, key):
        if key not in self.access_records:
            self.access_records[key] = ExpertAccessRecord()
        rec = self.access_records[key]
        if rec.access_count > 0:
            gap = self.current_token - rec.last_access
            rec.recent_gap = 0.7 * rec.recent_gap + 0.3 * gap  # EMA
        rec.last_access = self.current_token
        rec.access_count += 1
        self.max_freq = max(self.max_freq, rec.access_count)

    def record_routing(self, layer_idx, expert_indices):
        """Record routing decisions for offline Belady training."""
        if not hasattr(self, '_current_token_routing'):
            self._current_token_routing = []
        for eidx in expert_indices:
            self._current_token_routing.append((layer_idx, eidx))

    def end_token(self):
        """Call at end of each token to finalize trace."""
        if hasattr(self, '_current_token_routing'):
            self.routing_trace.append(self._current_token_routing)
            self._current_token_routing = []
        self.current_token += 1

    def trim(self):
        """Evict entries over budget using learned or LRU policy."""
        evicted = 0
        while len(self.cache) > self.budget:
            if self.use_learned and self.predictor.trained and len(self.cache) > 1:
                # Learned eviction: score all cached entries, evict highest
                victim = self._learned_evict()
            else:
                # Fallback to LRU
                victim = next(iter(self.cache))

            del self.cache[victim]
            evicted += 1
            self.eviction_count += 1
        return evicted

    def _learned_evict(self):
        """Use trained predictor to choose eviction victim."""
        keys = list(self.cache.keys())
        features = np.zeros((len(keys), 2), dtype=np.float32)

        for i, key in enumerate(keys):
            rec = self.access_records.get(key, ExpertAccessRecord())
            recency = self.current_token - rec.last_access
            features[i, 0] = 1.0 / max(recency, 1)  # inverse recency
            features[i, 1] = rec.access_count / max(self.max_freq, 1)  # normalized frequency

        scores = self.predictor.predict(features)
        victim_idx = np.argmax(scores)  # highest score = most evictable
        return keys[victim_idx]

    def reset_token_stats(self):
        self.token_hits = 0
        self.token_misses = 0

    @property
    def token_hit_rate(self):
        t = self.token_hits + self.token_misses
        return self.token_hits / t * 100 if t > 0 else 0

    @property
    def total_hit_rate(self):
        t = self.total_hits + self.total_misses
        return self.total_hits / t * 100 if t > 0 else 0


# ─── Belady Optimal Labeling (offline, needs full trace) ──────────

def compute_belady_labels(trace, cache_budget):
    """Given a full routing trace, compute Belady-optimal eviction labels.

    For each eviction event, labels which cached entry is optimal to evict
    (the one whose next use is furthest in the future).

    trace: list of list of (layer, expert) tuples per token
    returns: list of (features, label) training samples
    """
    # Build next-use index: for each (layer, expert), when is it next used?
    # Flatten trace with token indices
    all_accesses = []  # (token_idx, layer, expert)
    for t_idx, token_routing in enumerate(trace):
        for layer, expert in token_routing:
            all_accesses.append((t_idx, layer, expert))

    # Build forward next-use map
    next_use = {}  # (layer, expert) -> list of token indices (sorted)
    for t_idx, layer, expert in all_accesses:
        key = (layer, expert)
        if key not in next_use:
            next_use[key] = []
        next_use[key].append(t_idx)

    # Simulate cache with Belady labeling
    cache = OrderedDict()
    access_records = {}
    max_freq = 1
    samples = []

    for t_idx, token_routing in enumerate(trace):
        for layer, expert in token_routing:
            key = (layer, expert)

            # Update access tracking
            if key not in access_records:
                access_records[key] = {'count': 0, 'last': 0}
            access_records[key]['count'] += 1
            access_records[key]['last'] = t_idx
            max_freq = max(max_freq, access_records[key]['count'])

            if key in cache:
                cache.move_to_end(key)
                continue

            cache[key] = True

            if len(cache) > cache_budget:
                # Find Belady-optimal victim (furthest next use)
                cached_keys = list(cache.keys())
                best_victim = None
                best_next = -1

                for ck in cached_keys:
                    # Find next use after current token
                    uses = next_use.get(ck, [])
                    # Binary search for next use after t_idx
                    future = [u for u in uses if u > t_idx]
                    if not future:
                        best_victim = ck
                        break
                    elif future[0] > best_next:
                        best_next = future[0]
                        best_victim = ck

                if best_victim is None:
                    best_victim = cached_keys[0]

                # Generate training sample: features of ALL cached entries + label
                features_list = []
                for ck in cached_keys:
                    rec = access_records.get(ck, {'count': 0, 'last': 0})
                    recency = t_idx - rec['last']
                    features_list.append([
                        1.0 / max(recency, 1),
                        rec['count'] / max(max_freq, 1)
                    ])

                label_idx = cached_keys.index(best_victim)
                samples.append((np.array(features_list, dtype=np.float32), label_idx))

                del cache[best_victim]

    return samples


def train_predictor(samples, epochs=200, lr=1e-3):
    """Train a BeladyPredictor from Belady-labeled samples."""
    predictor = BeladyPredictor(hidden_dim=128)

    if not samples:
        print("  No training samples!")
        return predictor

    # Convert to pairwise training: for each eviction event,
    # the victim should have higher score than non-victims
    print(f"  Training on {len(samples)} eviction events...")

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for features, label_idx in samples:
            # Forward pass
            scores = predictor.predict(features)

            # Loss: victim should have highest score
            victim_score = scores[label_idx]

            # Margin loss: victim score should exceed all others
            for i in range(len(scores)):
                if i == label_idx:
                    continue
                margin = 1.0
                loss = max(0, margin - (victim_score - scores[i]))
                total_loss += loss

                if loss > 0:
                    # Gradient update (simplified SGD)
                    # Nudge weights to increase victim score, decrease other
                    _update_weights(predictor, features, label_idx, i, lr)

            if np.argmax(scores) == label_idx:
                correct += 1

        acc = correct / len(samples) * 100
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss:.1f} acc={acc:.1f}%")

    predictor.trained = True
    return predictor


def _update_weights(predictor, features, victim_idx, other_idx, lr):
    """Simple gradient step to push victim score up and other score down."""
    # Compute gradients via finite differences (simple but effective for tiny model)
    eps = 1e-4

    for param_name in ['w1', 'b1', 'w2', 'b2', 'w3', 'b3']:
        param = getattr(predictor, param_name)
        # Subsample for speed (don't compute full gradient)
        indices = np.random.choice(param.size, min(10, param.size), replace=False)
        flat = param.flatten()

        for idx in indices:
            old_val = flat[idx]

            flat[idx] = old_val + eps
            setattr(predictor, param_name, flat.reshape(param.shape))
            scores_plus = predictor.predict(features)

            flat[idx] = old_val - eps
            setattr(predictor, param_name, flat.reshape(param.shape))
            scores_minus = predictor.predict(features)

            # Gradient of (victim_score - other_score) w.r.t. param
            grad = ((scores_plus[victim_idx] - scores_plus[other_idx]) -
                    (scores_minus[victim_idx] - scores_minus[other_idx])) / (2 * eps)

            flat[idx] = old_val + lr * grad  # Ascent (maximize victim-other gap)
            setattr(predictor, param_name, flat.reshape(param.shape))


# ─── Model Forward Pass (same as v24 but with Belady cache) ──────

def get_moe(layer):
    if hasattr(layer, 'block_sparse_moe'):
        return layer.block_sparse_moe
    elif hasattr(layer, 'mlp'):
        return layer.mlp
    return None


def is_moe(layer):
    m = get_moe(layer)
    return m is not None and hasattr(m, 'gate') and hasattr(m, 'switch_mlp')


def streaming_moe_forward(moe_module, x, expert_cache, use_cache=True):
    B, S, H = x.shape

    gates_out = moe_module.gate(x)
    if isinstance(gates_out, tuple):
        inds, scores = gates_out
    else:
        k = moe_module.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gates_out, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates_out, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)
    mx.eval(inds, scores)

    topk = inds.shape[-1]
    mlp = moe_module.switch_mlp
    gs_val = mlp.gate_proj.group_size
    bits = mlp.gate_proj.bits
    layer_idx = getattr(moe_module, '_ef_layer_idx', -1)

    shared = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared = moe_module.shared_experts(x)
        mx.eval(shared)

    inds_list = inds.reshape(B * S, topk).tolist()
    scores_flat = scores.reshape(B * S, topk)
    x_flat = x.reshape(B * S, H)

    # Record routing for Belady trace
    needed = set()
    for t in range(B * S):
        for k_i in range(topk):
            eidx = inds_list[t][k_i]
            needed.add(eidx)
            expert_cache.record_routing(layer_idx, [eidx])

    expert_weights = {}
    miss_parts = []
    for eidx in needed:
        cache_key = (layer_idx, eidx)
        cached = expert_cache.get(cache_key) if use_cache else None

        if cached is not None:
            expert_weights[eidx] = cached
        else:
            parts = []
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(mlp, proj_name)
                w = proj["weight"][eidx]
                s = proj["scales"][eidx]
                b_arr = proj.get("biases")
                b = b_arr[eidx] if b_arr is not None else None
                parts.extend([w, s, b])
                miss_parts.extend([p for p in [w, s, b] if p is not None])
            expert_weights[eidx] = tuple(parts)
            if use_cache:
                expert_cache.put(cache_key, tuple(parts))

    if miss_parts:
        mx.eval(*miss_parts)

    with mx.stream(mx.cpu):
        token_outs = []
        for t in range(B * S):
            x_t = x_flat[t:t+1]
            expert_results = []
            for k_i in range(topk):
                eidx = inds_list[t][k_i]
                score = scores_flat[t, k_i]
                gw, gs, gb, uw, us, ub, dw, ds, db = expert_weights[eidx]
                g = mx.quantized_matmul(x_t, gw, gs, gb, transpose=True, group_size=gs_val, bits=bits)
                u = mx.quantized_matmul(x_t, uw, us, ub, transpose=True, group_size=gs_val, bits=bits)
                out = mx.quantized_matmul(nn.silu(g) * u, dw, ds, db, transpose=True, group_size=gs_val, bits=bits)
                expert_results.append(out * score)
            combined = expert_results[0]
            for er in expert_results[1:]:
                combined = combined + er
            token_outs.append(combined)
        routed = mx.concatenate(token_outs, axis=0).reshape(B, S, H)
        result = (routed + shared) if shared is not None else routed
        mx.eval(result)

    return result


def create_mask(seq_len, cache_offset):
    if seq_len == 1:
        return None
    from mlx_lm.models.base import create_causal_mask
    return create_causal_mask(seq_len, cache_offset)


def generate(model, tokenizer, prompt, max_tokens, cache_budget=300,
             predictor_path=None, trace_mode=False, verbose=True):
    from mlx_lm.models.cache import KVCache

    input_ids = tokenizer.encode(prompt)
    num_layers = len(model.model.layers)
    kv_caches = [KVCache() for _ in range(num_layers)]

    moe_indices = []
    for i, layer in enumerate(model.model.layers):
        if is_moe(layer):
            moe_indices.append(i)
            get_moe(layer)._ef_layer_idx = i

    expert_cache = BeladyExpertCache(
        budget=cache_budget,
        use_learned=(not trace_mode and predictor_path is not None)
    )

    # Load trained predictor if available
    if predictor_path and os.path.exists(predictor_path):
        expert_cache.predictor.load(predictor_path)
        if verbose:
            print(f"  Loaded trained predictor from {predictor_path}")

    generated_ids = []
    token_times = []

    if verbose:
        mode_str = "TRACE" if trace_mode else ("BELADY" if expert_cache.use_learned else "LRU")
        print(f"\n  Prompt: {prompt!r} ({len(input_ids)} tokens)")
        print(f"  Layers: {num_layers} ({len(moe_indices)} MoE)")
        print(f"  Cache: {cache_budget} experts (~{cache_budget * 11 / 1024:.1f}GB)")
        print(f"  Eviction: {mode_str}")

    for step in range(max_tokens):
        t_token = time.time()
        expert_cache.reset_token_stats()

        if step == 0:
            ids = mx.array([input_ids])
        else:
            ids = mx.array([[generated_ids[-1]]])

        x = model.model.embed_tokens(ids)
        mx.eval(x)

        seq_len = ids.shape[1]
        cache_offset = kv_caches[0].offset if not kv_caches[0].empty() else 0
        mask = create_mask(seq_len, cache_offset)

        attn_time = 0
        moe_time = 0

        for i, layer in enumerate(model.model.layers):
            t_a = time.time()
            h = layer.input_layernorm(x)
            h = layer.self_attn(h, mask, kv_caches[i])
            x = x + h
            mx.eval(x)
            attn_time += time.time() - t_a

            t_m = time.time()
            h = layer.post_attention_layernorm(x)
            moe_mod = get_moe(layer)
            if i in moe_indices:
                h = streaming_moe_forward(moe_mod, h, expert_cache, use_cache=(step > 0))
            else:
                h = moe_mod(h)
                mx.eval(h)
            x = x + h
            mx.eval(x)
            moe_time += time.time() - t_m

        evicted = expert_cache.trim()
        expert_cache.end_token()

        x = model.model.norm(x[:, -1:, :])
        logits = model.lm_head(x)
        mx.eval(logits)

        next_id = int(mx.argmax(logits[0, 0]).item())
        generated_ids.append(next_id)

        dt = time.time() - t_token
        token_times.append(dt)

        if verbose:
            try:
                text = tokenizer.decode([next_id])
            except:
                text = f"[{next_id}]"
            mode = "prefill" if step == 0 else "decode"
            print(f"  T{step+1}({mode}): {text!r} | "
                  f"{dt:.1f}s ({1/dt:.3f} t/s) | "
                  f"a={attn_time:.1f} m={moe_time:.1f} | "
                  f"hit {expert_cache.token_hit_rate:.0f}% "
                  f"(cache: {len(expert_cache.cache)}, evict: {evicted})", flush=True)

    try:
        output_text = tokenizer.decode(generated_ids)
    except:
        output_text = str(generated_ids)

    prefill_time = token_times[0] if token_times else 0
    decode_times = token_times[1:] if len(token_times) > 1 else []
    steady = token_times[-5:] if len(token_times) >= 6 else decode_times
    steady_avg = sum(steady) / len(steady) if steady else 0

    return {
        "prompt": prompt,
        "output": output_text,
        "tokens": len(generated_ids),
        "prefill_s": round(prefill_time, 2),
        "decode_avg_s": round(sum(decode_times)/len(decode_times), 2) if decode_times else 0,
        "steady_avg_s": round(steady_avg, 2),
        "steady_tok_s": round(1/steady_avg, 4) if steady_avg > 0 else 0,
        "total_s": round(sum(token_times), 1),
        "cache_hit_rate": round(expert_cache.total_hit_rate, 1),
        "eviction_policy": "belady" if expert_cache.use_learned and expert_cache.predictor.trained else "lru",
        "routing_trace_len": len(expert_cache.routing_trace),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=15)
    p.add_argument("--cache-budget", type=int, default=300)
    p.add_argument("--predictor-path", default=os.path.expanduser(
        "~/dev/expertflow/experiments/belady_predictor.npz"))
    p.add_argument("--trace-mode", action="store_true",
                   help="Collect routing trace (no learned eviction)")
    p.add_argument("--train", action="store_true",
                   help="Train predictor from collected trace, then run with it")
    args = p.parse_args()

    print("=" * 60)
    print("  ExpertFlow v25 — Belady-Approximate Cache")
    print("=" * 60)
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Free:  {free_gb():.1f} GB")

    mx.set_memory_limit(int(100 * 1024**3))
    mx.set_cache_limit(int(4 * 1024**3))
    try:
        mx.set_wired_limit(int(60 * 1024**3))
    except:
        pass

    import mlx_lm
    print("  Loading (lazy)...", flush=True)
    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    print(f"  Loaded. Free: {free_gb():.0f}G")

    if args.train:
        # Phase 1: Collect trace with LRU
        print("\n--- Phase 1: Collecting routing trace (LRU mode) ---")
        try:
            results = generate(model, tokenizer, args.prompt, args.max_tokens,
                              cache_budget=args.cache_budget, trace_mode=True)
            print(f"  Trace collected: {results['routing_trace_len']} tokens")
            print(f"  LRU hit rate: {results['cache_hit_rate']}%")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            return

        # Save trace for analysis
        trace_path = os.path.expanduser("~/dev/expertflow/experiments/routing_trace.json")
        # Note: trace is embedded in cache object, would need to be extracted
        # For now, skip trace save and use synthetic training

        print("\n--- Phase 2: Training would happen here ---")
        print("  (Full training requires routing trace data)")
        print("  Skipping to LRU baseline for now")
    else:
        # Run with existing predictor or LRU fallback
        try:
            results = generate(model, tokenizer, args.prompt, args.max_tokens,
                              cache_budget=args.cache_budget,
                              predictor_path=args.predictor_path if not args.trace_mode else None,
                              trace_mode=args.trace_mode)
        except Exception as e:
            print(f"\n  FAILED: {e}")
            traceback.print_exc()
            return

    print(f"\n{'='*60}")
    print(f"  OUTPUT: {results['prompt']}{results['output']}")
    print(f"  PREFILL: {results['prefill_s']}s")
    print(f"  STEADY:  {results['steady_avg_s']}s/tok ({results['steady_tok_s']} tok/s)")
    print(f"  Cache: {results['cache_hit_rate']}% hit ({results['eviction_policy']})")
    print(f"  Total: {results['total_s']}s for {results['tokens']} tokens")

    outfile = os.path.expanduser(f"~/dev/expertflow/experiments/v25_{time.strftime('%H%M%S')}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
