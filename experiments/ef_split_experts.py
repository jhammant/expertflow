#!/usr/bin/env python3
"""
ExpertFlow — Expert Weight Splitter
====================================
Restructures MoE model weights for sequential I/O.

Problem: Expert weights stored as stacked tensor [160, out_dim, packed_in].
Accessing expert_idx requires random mmap page faults into a ~600MB tensor.
Each page fault = syscall + potential page eviction.

Solution: Split into per-layer files with experts stored sequentially.
Layout: experts/{layer_idx}.bin
  - Header: [n_experts, topk, n_projections, expert_size_bytes, ...]
  - Per expert: [gate_proj_weight, gate_proj_scales, gate_proj_biases,
                 up_proj_weight, up_proj_scales, up_proj_biases,
                 down_proj_weight, down_proj_scales, down_proj_biases]

Benefits:
  - Sequential read: loading expert K reads contiguous bytes
  - Page-aligned: each expert can be at a page boundary
  - OS can prefetch sequentially (readahead)
  - madvise(MADV_SEQUENTIAL) possible on individual expert regions
"""

import os, sys, time, json, struct
import numpy as np
os.environ["MLX_LAZY_INITIALIZATION"] = "1"


def split_experts(model_path, output_dir, verbose=True):
    """
    Split stacked expert weights into per-layer binary files.
    Each file contains all expert data for one MoE layer, laid out
    so that loading expert K is a single sequential read.
    """
    import mlx.core as mx
    import mlx_lm

    if verbose:
        print(f"  Loading model from {model_path}...", flush=True)
    model, _ = mlx_lm.load(model_path, lazy=True)

    os.makedirs(output_dir, exist_ok=True)

    # Metadata for loading
    metadata = {
        "model_path": model_path,
        "layers": {},
    }

    moe_count = 0
    total_bytes = 0

    for layer_idx, layer in enumerate(model.model.layers):
        # Find MoE module
        moe = None
        if hasattr(layer, 'block_sparse_moe'):
            moe = layer.block_sparse_moe
        elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'switch_mlp'):
            moe = layer.mlp

        if moe is None or not hasattr(moe, 'switch_mlp'):
            continue

        mlp = moe.switch_mlp
        moe_count += 1

        # Get tensor shapes
        n_experts = mlp.gate_proj["weight"].shape[0]
        projections = ['gate_proj', 'up_proj', 'down_proj']

        # Calculate per-expert data size
        expert_parts = []  # (proj_name, array_name, shape_per_expert)
        expert_byte_size = 0

        for proj_name in projections:
            proj = getattr(mlp, proj_name)
            for arr_name in ['weight', 'scales']:
                arr = proj[arr_name]
                per_expert_shape = arr.shape[1:]  # Remove expert dim
                per_expert_bytes = int(np.prod(per_expert_shape)) * arr.dtype.size
                expert_parts.append((proj_name, arr_name, per_expert_shape, arr.dtype, per_expert_bytes))
                expert_byte_size += per_expert_bytes

            # Biases (optional)
            biases = proj.get("biases")
            if biases is not None:
                per_expert_shape = biases.shape[1:]
                per_expert_bytes = int(np.prod(per_expert_shape)) * biases.dtype.size
                expert_parts.append((proj_name, 'biases', per_expert_shape, biases.dtype, per_expert_bytes))
                expert_byte_size += per_expert_bytes

        # Page-align expert size (16KB pages on macOS)
        PAGE_SIZE = 16384
        aligned_expert_size = ((expert_byte_size + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE

        # Write per-layer file
        layer_file = os.path.join(output_dir, f"layer_{layer_idx:03d}.bin")

        # Header: magic, version, n_experts, expert_data_size, aligned_size, n_parts
        MAGIC = b'EFEX'  # ExpertFlow EXperts
        VERSION = 1

        header = struct.pack('<4sIIIII',
            MAGIC, VERSION, n_experts, expert_byte_size,
            aligned_expert_size, len(expert_parts))

        # Part descriptors
        part_descs = []
        offset_in_expert = 0
        for proj_name, arr_name, shape, dtype, nbytes in expert_parts:
            # Encode dtype
            dtype_map = {mx.uint32: 0, mx.float16: 1, mx.float32: 2, mx.bfloat16: 3}
            dtype_id = dtype_map.get(dtype, 0)
            desc = struct.pack('<32s32sIIII',
                proj_name.encode()[:32].ljust(32, b'\0'),
                arr_name.encode()[:32].ljust(32, b'\0'),
                dtype_id, nbytes, offset_in_expert,
                len(shape))
            # Shape dims
            for d in shape:
                desc += struct.pack('<I', d)
            # Pad to fixed size
            desc = desc.ljust(128, b'\0')
            part_descs.append(desc)
            offset_in_expert += nbytes

        header_total = header + b''.join(part_descs)
        # Pad header to page alignment
        header_padded = header_total.ljust(
            ((len(header_total) + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE, b'\0'
        )
        data_start = len(header_padded)

        with open(layer_file, 'wb') as f:
            f.write(header_padded)

            for eidx in range(n_experts):
                expert_data = bytearray()

                for proj_name, arr_name, shape, dtype, nbytes in expert_parts:
                    proj = getattr(mlp, proj_name)
                    if arr_name == 'biases':
                        arr = proj.get("biases")
                    else:
                        arr = proj[arr_name]

                    # Extract this expert's slice
                    slice_data = arr[eidx]
                    mx.eval(slice_data)
                    expert_data.extend(bytes(memoryview(np.array(slice_data))))

                # Pad to aligned size
                expert_data.extend(b'\0' * (aligned_expert_size - len(expert_data)))
                f.write(bytes(expert_data))

                if verbose and eidx == 0:
                    print(f"  Layer {layer_idx}: {n_experts} experts × "
                          f"{expert_byte_size/1024:.0f}KB = "
                          f"{n_experts * aligned_expert_size / 1024 / 1024:.1f}MB", flush=True)

        file_size = os.path.getsize(layer_file)
        total_bytes += file_size

        metadata["layers"][str(layer_idx)] = {
            "file": f"layer_{layer_idx:03d}.bin",
            "n_experts": n_experts,
            "expert_byte_size": expert_byte_size,
            "aligned_expert_size": aligned_expert_size,
            "data_start": data_start,
            "parts": [
                {"proj": p, "arr": a, "shape": list(s), "dtype": d.name if hasattr(d, 'name') else str(d), "bytes": b}
                for p, a, s, d, b in expert_parts
            ],
            "group_size": mlp.gate_proj.group_size,
            "bits": mlp.gate_proj.bits,
        }

    # Save metadata
    meta_file = os.path.join(output_dir, "metadata.json")
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\n  Split {moe_count} MoE layers")
        print(f"  Total: {total_bytes / 1024 / 1024 / 1024:.2f} GB")
        print(f"  Saved to: {output_dir}")

    return metadata


def main():
    import argparse
    p = argparse.ArgumentParser(description="Split MoE expert weights for sequential I/O")
    p.add_argument("--model", default=os.path.expanduser("~/models/glm-4.5-4bit"))
    p.add_argument("--output", default=None, help="Output directory (default: model_path/experts/)")
    args = p.parse_args()

    output_dir = args.output or os.path.join(args.model, "experts")

    print("=" * 60)
    print("  ExpertFlow — Expert Weight Splitter")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Output: {output_dir}")

    t0 = time.time()
    metadata = split_experts(args.model, output_dir)
    print(f"\n  Done in {time.time()-t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
