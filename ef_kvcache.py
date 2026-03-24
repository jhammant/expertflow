#!/usr/bin/env python3
"""
ExpertFlow v3 — KV Cache + Expert Cache + Wired Memory
======================================================
Token 1: Full forward pass (slow, ~70s)
Token 2+: Only process NEW token using cached K/V (10-50x faster)

Plus expert caching with wired memory for MoE speedup.
"""

import os, sys, time, json, subprocess, math
os.environ["MLX_LAZY_INITIALIZATION"] = "1"

import mlx.core as mx
import mlx.nn as nn
import gc
from collections import OrderedDict

# ═══ Config ═══
EXPERT_CACHE_GB = 30
WIRED_GB = 90

def free_gb():
    try:
        out = subprocess.check_output(["vm_stat"], timeout=2).decode()
        f = i = 0
        for l in out.split("\n"):
            if "Pages free" in l: f = int(l.split()[-1].rstrip("."))
            elif "Pages inactive" in l: i = int(l.split()[-1].rstrip("."))
        return (f + i) * 16384 / 1e9
    except:
        return 999

# ═══ Expert Cache ═══
class ExpertCache:
    def __init__(self, budget_gb):
        self.cache = OrderedDict()
        self.budget = int(budget_gb * 1024**3)
        self.size = 0
        self.hits = self.misses = 0
        self.ENTRY = 84 * 1024**2  # ~84MB per expert set
    
    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key, value):
        while self.size + self.ENTRY > self.budget and self.cache:
            self.cache.popitem(last=False)
            self.size -= self.ENTRY
        self.cache[key] = value
        self.size += self.ENTRY
    
    def stats(self):
        total = self.hits + self.misses
        rate = self.hits / max(total, 1) * 100
        return f"{len(self.cache)} experts cached, {rate:.0f}% hit rate"

expert_cache = ExpertCache(EXPERT_CACHE_GB)

# ═══ KV Cache ═══
class KVCache:
    """Per-layer KV cache for incremental decoding."""
    def __init__(self, num_layers):
        self.keys = [None] * num_layers    # [batch, heads, seq, head_dim]
        self.values = [None] * num_layers
        self.seq_len = 0
    
    def update(self, layer_idx, k, v):
        """Append new K/V to cache."""
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = k
            self.values[layer_idx] = v
        else:
            self.keys[layer_idx] = mx.concatenate([self.keys[layer_idx], k], axis=2)
            self.values[layer_idx] = mx.concatenate([self.values[layer_idx], v], axis=2)
        mx.eval(self.keys[layer_idx], self.values[layer_idx])
    
    def get(self, layer_idx):
        return self.keys[layer_idx], self.values[layer_idx]

# ═══ Expert Loading ═══
def dequant_on_cpu(proj, eidx):
    with mx.stream(mx.cpu):
        w = proj["weight"][eidx]
        s = proj["scales"][eidx]
        b = proj.get("biases")
        b = b[eidx] if b is not None else None
        result = mx.dequantize(w, s, b, group_size=proj.group_size, bits=proj.bits)
        mx.eval(result)
    return result

def get_expert(mlp, layer_idx, eidx):
    """Get expert weights with caching."""
    key = (layer_idx, eidx)
    cached = expert_cache.get(key)
    if cached is not None:
        return cached
    
    gw = dequant_on_cpu(mlp.gate_proj, eidx)
    uw = dequant_on_cpu(mlp.up_proj, eidx)
    dw = dequant_on_cpu(mlp.down_proj, eidx)
    
    expert_cache.put(key, (gw, uw, dw))
    return (gw, uw, dw)

# ═══ Custom Attention with KV Cache ═══
def attention_with_kv(attn_module, x, layer_idx, kv_cache, mask=None):
    """
    Run attention with KV caching.
    First call: compute full K/V, cache them.
    Subsequent calls: only compute K/V for new token, append to cache.
    """
    B, S, _ = x.shape
    
    # Project Q, K, V
    # MLX attention modules have different structures, handle generically
    if hasattr(attn_module, 'q_proj'):
        q = attn_module.q_proj(x)
        k = attn_module.k_proj(x)
        v = attn_module.v_proj(x)
    else:
        # Fallback — just call the module directly
        return attn_module(x, mask=mask)
    
    # Reshape for multi-head attention
    num_heads = attn_module.n_heads if hasattr(attn_module, 'n_heads') else attn_module.num_heads
    num_kv_heads = getattr(attn_module, 'n_kv_heads', getattr(attn_module, 'num_kv_heads', num_heads))
    head_dim = q.shape[-1] // num_heads
    
    q = q.reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    
    # Apply RoPE if the module has it
    if hasattr(attn_module, 'rope'):
        offset = kv_cache.seq_len if kv_cache.keys[layer_idx] is not None else 0
        q = attn_module.rope(q, offset=offset)
        k = attn_module.rope(k, offset=offset)
    
    # Update KV cache
    kv_cache.update(layer_idx, k, v)
    
    # Get full cached K/V
    full_k, full_v = kv_cache.get(layer_idx)
    
    # GQA: expand K/V heads if needed
    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        full_k = mx.repeat(full_k, repeat_factor, axis=1)
        full_v = mx.repeat(full_v, repeat_factor, axis=1)
    
    # Scaled dot-product attention
    scale = math.sqrt(head_dim)
    scores = (q @ full_k.transpose(0, 1, 3, 2)) / scale
    
    # Causal mask
    total_len = full_k.shape[2]
    if S == 1 and total_len > 1:
        # Decoding: new token can attend to all cached tokens
        pass
    elif total_len > 1:
        # Prefill: causal mask
        causal = mx.triu(mx.full((S, total_len), float('-inf')), k=total_len - S + 1)
        scores = scores + causal
    
    weights = mx.softmax(scores, axis=-1)
    output = weights @ full_v
    
    # Reshape back
    output = output.transpose(0, 2, 1, 3).reshape(B, S, -1)
    
    # Output projection
    if hasattr(attn_module, 'o_proj'):
        output = attn_module.o_proj(output)
    
    mx.eval(output)
    return output

# ═══ MoE Forward ═══
def moe_forward(moe_module, x, layer_idx):
    B, S, H = x.shape
    
    inds, scores = moe_module.gate(x)
    mx.eval(inds, scores)
    
    topk = inds.shape[-1]
    mlp = moe_module.switch_mlp
    
    # Shared experts
    shared = None
    if hasattr(moe_module, 'shared_experts') and moe_module.shared_experts is not None:
        shared = moe_module.shared_experts(x)
        mx.eval(shared)
    
    x_flat = x.reshape(B * S, H)
    inds_2d = inds.reshape(B * S, topk)
    scores_2d = scores.reshape(B * S, topk)
    
    token_outs = []
    for t in range(B * S):
        x_t = x_flat[t:t+1]
        out = mx.zeros((1, H))
        
        for k in range(topk):
            eidx = int(inds_2d[t, k].item())
            score = scores_2d[t, k]
            
            gw, uw, dw = get_expert(mlp, layer_idx, eidx)
            
            g = x_t @ gw.T
            u = x_t @ uw.T
            out = out + (nn.silu(g) * u) @ dw.T * score
        
        mx.eval(out)
        token_outs.append(out)
    
    routed = mx.concatenate(token_outs, axis=0).reshape(B, S, H)
    result = (routed + shared) if shared is not None else routed
    mx.eval(result)
    mx.clear_cache()
    
    return result

# ═══ Generation ═══
def generate(model, tokenizer, prompt, max_tokens):
    input_ids = tokenizer.encode(prompt)
    total_layers = len(model.model.layers)
    generated = []
    token_times = []
    
    kv = KVCache(total_layers)
    
    print(f"\n  Prompt: {prompt!r} ({len(input_ids)} tokens)")
    print(f"  Layers: {total_layers}")
    print(f"  Expert cache: {EXPERT_CACHE_GB}GB", flush=True)
    
    for step in range(max_tokens):
        t0 = time.time()
        
        if step == 0:
            # Prefill: process entire prompt
            ids = input_ids
        else:
            # Decode: only process new token
            ids = [generated[-1]]
        
        x = model.model.embed_tokens(mx.array([ids]))
        mx.eval(x)
        
        for i, layer in enumerate(model.model.layers):
            # Attention with KV cache
            h = layer.input_layernorm(x)
            mx.eval(h)
            
            try:
                h = attention_with_kv(layer.self_attn, h, i, kv)
            except Exception as e:
                # Fallback to standard attention (no KV cache benefit)
                h = layer.self_attn(h)
                mx.eval(h)
            
            x = x + h
            mx.eval(x)
            
            # MLP/MoE
            h = layer.post_attention_layernorm(x)
            mx.eval(h)
            
            if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'switch_mlp'):
                h = moe_forward(layer.mlp, h, i)
            else:
                h = layer.mlp(h)
                mx.eval(h)
            
            x = x + h
            mx.eval(x)
            
            if (i+1) % 10 == 0:
                f = free_gb()
                print(f"[L{i+1}/{total_layers} {f:.0f}G]", end=" ", flush=True)
        
        # Update sequence length
        kv.seq_len += len(ids)
        
        # Logits for last token only
        x_last = model.model.norm(x[:, -1:, :])
        logits = model.lm_head(x_last)
        mx.eval(logits)
        
        next_id = int(mx.argmax(logits[0, 0]).item())
        generated.append(next_id)
        
        dt = time.time() - t0
        token_times.append(dt)
        
        try:
            text = tokenizer.decode([next_id])
        except:
            text = f"[{next_id}]"
        
        mode = "prefill" if step == 0 else "decode"
        print(f"\n  ✅ Token {step+1} ({mode}): {text!r} ({dt:.1f}s | {1/dt:.3f} tok/s)", flush=True)
        
        if hasattr(tokenizer, 'eos_token_id') and next_id == tokenizer.eos_token_id:
            break
    
    try:
        full = tokenizer.decode(generated)
    except:
        full = str(generated)
    
    # Separate prefill vs decode speeds
    prefill_time = token_times[0] if token_times else 0
    decode_times = token_times[1:] if len(token_times) > 1 else []
    decode_avg = sum(decode_times) / len(decode_times) if decode_times else 0
    
    return {
        "prompt": prompt,
        "output": full,
        "full_text": prompt + full,
        "tokens": len(generated),
        "prefill_s": round(prefill_time, 1),
        "prefill_tok_s": round(1/prefill_time, 4) if prefill_time > 0 else 0,
        "decode_avg_s": round(decode_avg, 1),
        "decode_tok_s": round(1/decode_avg, 4) if decode_avg > 0 else 0,
        "times": [round(t, 2) for t in token_times],
        "total_s": round(sum(token_times), 1),
        "expert_cache": expert_cache.stats(),
    }

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=10)
    args = p.parse_args()
    
    print("=" * 60)
    print("  ExpertFlow v3 — KV Cache + Expert Cache + Wired Memory")
    print("=" * 60)
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Free: {free_gb():.1f}GB")
    
    mx.set_memory_limit(int(110 * 1024**3))
    mx.set_cache_limit(int(15 * 1024**3))
    try:
        mx.set_wired_limit(int(WIRED_GB * 1024**3))
        print(f"  Wired: {WIRED_GB}GB")
    except:
        pass
    
    import mlx_lm
    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    
    results = generate(model, tokenizer, args.prompt, args.max_tokens)
    
    print(f"\n{'=' * 60}")
    print(f"  OUTPUT: {results['full_text']}")
    print(f"  PREFILL: {results['prefill_s']}s ({results['prefill_tok_s']} tok/s)")
    print(f"  DECODE:  {results['decode_avg_s']}s ({results['decode_tok_s']} tok/s)")
    print(f"  TOTAL:   {results['total_s']}s for {results['tokens']} tokens")
    print(f"  CACHE:   {results['expert_cache']}")
    
    outfile = os.path.expanduser(f"~/dev/expertflow/kv_result_{time.strftime('%H%M%S')}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {outfile}")

if __name__ == "__main__":
    main()
