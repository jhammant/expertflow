#!/usr/bin/env python3
"""
MLX Python Bridge for ExpertFlow

Communicates with Rust via JSON-over-stdio protocol.
Handles MLX model loading, expert state tracking, and inference.

Protocol:
  - Input: One JSON object per line on stdin
  - Output: One JSON response per line on stdout
  - Errors: JSON with {"error": "message"} on stdout

Commands:
  - load_model: {"cmd": "load_model", "path": str, "model_type": str}
  - get_expert_state: {"cmd": "get_expert_state", "expert_id": int}
  - prefetch_expert: {"cmd": "prefetch_expert", "expert_id": int}
  - evict_expert: {"cmd": "evict_expert", "expert_id": int}
  - router_predict: {"cmd": "router_predict", "layer": int, "hidden_states": [float]}
  - generate_token: {"cmd": "generate_token", "input_ids": [int]}
  - shutdown: {"cmd": "shutdown"}
"""

import sys
import json
import logging
from typing import Dict, List, Optional, Any

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as mlx_load
    from mlx_lm import generate as mlx_generate
except ImportError:
    # Graceful degradation if MLX not installed
    mx = None
    nn = None
    mlx_load = None
    mlx_generate = None

# Configure logging to stderr (stdout is for JSON protocol)
logging.basicConfig(
    level=logging.INFO,
    format='[MLX Bridge] %(levelname)s: %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class MLXModelBridge:
    """Manages MLX model state and expert loading."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path: Optional[str] = None
        self.model_type: Optional[str] = None
        self.expert_cache: Dict[int, Any] = {}
        self.loaded_experts: set = set()

    def load_model(self, path: str, model_type: str = "auto") -> Dict[str, Any]:
        """
        Load an MLX model from HuggingFace or local path.

        Args:
            path: HuggingFace model ID (e.g., "mlx-community/DeepSeek-V3-Base-4bit")
                  or local path to MLX model
            model_type: Model architecture type (auto-detected if "auto")

        Returns:
            {"status": "ok", "num_experts": int, "num_layers": int}
        """
        if mx is None:
            return {"error": "MLX not installed. Install via: pip install mlx mlx-lm"}

        try:
            logger.info(f"Loading MLX model from: {path}")
            self.model, self.tokenizer = mlx_load(path)
            self.model_path = path
            self.model_type = model_type

            # Introspect model to count experts and layers
            num_experts = 0
            num_layers = 0

            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                num_layers = len(self.model.model.layers)

                # Check first layer for MoE structure
                if num_layers > 0:
                    first_layer = self.model.model.layers[0]
                    if hasattr(first_layer, 'block_sparse_moe'):
                        # DeepSeek-V3 style
                        num_experts = len(first_layer.block_sparse_moe.experts)
                    elif hasattr(first_layer, 'mlp') and hasattr(first_layer.mlp, 'experts'):
                        # Qwen style
                        num_experts = len(first_layer.mlp.experts)

            logger.info(f"Loaded model: {num_layers} layers, {num_experts} experts/layer")

            return {
                "status": "ok",
                "num_experts": num_experts,
                "num_layers": num_layers,
                "model_type": self.model_type
            }

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {"error": f"Model load failed: {str(e)}"}

    def get_expert_state(self, expert_id: int) -> Dict[str, Any]:
        """
        Get the state of a specific expert.

        Args:
            expert_id: Global expert ID (layer * num_experts_per_layer + expert_idx)

        Returns:
            {"status": "loaded"|"evicted", "memory_bytes": int}
        """
        if self.model is None:
            return {"error": "No model loaded"}

        state = "loaded" if expert_id in self.loaded_experts else "evicted"
        memory_bytes = 0

        # Estimate memory usage if expert is cached
        if expert_id in self.expert_cache:
            # Rough estimate: sum of parameter sizes
            expert_params = self.expert_cache[expert_id]
            if hasattr(expert_params, 'nbytes'):
                memory_bytes = expert_params.nbytes

        return {
            "status": state,
            "expert_id": expert_id,
            "memory_bytes": memory_bytes
        }

    def prefetch_expert(self, expert_id: int) -> Dict[str, Any]:
        """
        Prefetch expert weights into memory (no-op in MLX, weights are lazy-loaded).

        Args:
            expert_id: Global expert ID

        Returns:
            {"status": "ok"}
        """
        if self.model is None:
            return {"error": "No model loaded"}

        # MLX lazy-loads weights, so we just mark as loaded
        self.loaded_experts.add(expert_id)
        logger.debug(f"Prefetched expert {expert_id}")

        return {"status": "ok", "expert_id": expert_id}

    def evict_expert(self, expert_id: int) -> Dict[str, Any]:
        """
        Evict expert weights from memory.

        Args:
            expert_id: Global expert ID

        Returns:
            {"status": "ok"}
        """
        if self.model is None:
            return {"error": "No model loaded"}

        # Remove from cache and loaded set
        self.expert_cache.pop(expert_id, None)
        self.loaded_experts.discard(expert_id)
        logger.debug(f"Evicted expert {expert_id}")

        return {"status": "ok", "expert_id": expert_id}

    def router_predict(self, layer: int, hidden_states: List[float]) -> Dict[str, Any]:
        """
        Run router network to predict expert selection.

        Args:
            layer: Layer index
            hidden_states: Input hidden states (flattened)

        Returns:
            {"expert_ids": [int], "weights": [float]}
        """
        if self.model is None:
            return {"error": "No model loaded"}

        try:
            # For now, return placeholder (requires implementing router forward pass)
            # In production, would run: router_logits = model.layers[layer].router(hidden_states)
            logger.warning("router_predict not yet implemented, returning placeholder")

            return {
                "expert_ids": [0, 1, 2],  # Placeholder: top-3 experts
                "weights": [0.5, 0.3, 0.2]
            }

        except Exception as e:
            logger.error(f"Router prediction failed: {e}")
            return {"error": f"Router prediction failed: {str(e)}"}

    def generate_token(self, input_ids: List[int]) -> Dict[str, Any]:
        """
        Generate next token given input IDs.

        Args:
            input_ids: Token IDs

        Returns:
            {"token_id": int, "text": str}
        """
        if self.model is None or self.tokenizer is None:
            return {"error": "No model loaded"}

        try:
            # Convert to MLX array
            input_array = mx.array([input_ids])

            # Forward pass (simplified - mlx-lm has better generation utilities)
            logits = self.model(input_array)
            next_token_id = int(mx.argmax(logits[0, -1, :]))

            # Decode token
            text = self.tokenizer.decode([next_token_id])

            return {
                "token_id": next_token_id,
                "text": text
            }

        except Exception as e:
            logger.error(f"Token generation failed: {e}")
            return {"error": f"Token generation failed: {str(e)}"}


def main():
    """Main event loop: read JSON commands from stdin, write responses to stdout."""
    bridge = MLXModelBridge()

    logger.info("MLX Bridge started, waiting for commands...")

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                cmd = json.loads(line)
            except json.JSONDecodeError as e:
                response = {"error": f"Invalid JSON: {str(e)}"}
                print(json.dumps(response), flush=True)
                continue

            # Dispatch command
            cmd_type = cmd.get("cmd")

            if cmd_type == "load_model":
                response = bridge.load_model(
                    cmd.get("path", ""),
                    cmd.get("model_type", "auto")
                )

            elif cmd_type == "get_expert_state":
                response = bridge.get_expert_state(cmd.get("expert_id", 0))

            elif cmd_type == "prefetch_expert":
                response = bridge.prefetch_expert(cmd.get("expert_id", 0))

            elif cmd_type == "evict_expert":
                response = bridge.evict_expert(cmd.get("expert_id", 0))

            elif cmd_type == "router_predict":
                response = bridge.router_predict(
                    cmd.get("layer", 0),
                    cmd.get("hidden_states", [])
                )

            elif cmd_type == "generate_token":
                response = bridge.generate_token(cmd.get("input_ids", []))

            elif cmd_type == "shutdown":
                logger.info("Shutdown requested")
                response = {"status": "ok"}
                print(json.dumps(response), flush=True)
                break

            else:
                response = {"error": f"Unknown command: {cmd_type}"}

            # Send response
            print(json.dumps(response), flush=True)

    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
