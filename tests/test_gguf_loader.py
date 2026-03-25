"""Tests for GGUF parser and expert mmap loader edge cases.

Covers: missing files, corrupt headers, unsupported quant types,
tensor byte size calculations, split GGUF parsing, and dequantization.
"""

import struct
import tempfile
import os

import numpy as np
import pytest

from ef_deepseek_mmap import (
    GGUF_MAGIC,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_IQ1_S,
    GGML_BLOCK_SIZES,
    QK_K,
    tensor_byte_size,
    parse_gguf,
    parse_split_gguf,
    dequant_q2_k,
    dequant_q2_k_fast,
    GGUFTensorInfo,
)


class TestTensorByteSize:
    """Tests for tensor_byte_size calculation."""

    def test_f32_scalar(self):
        assert tensor_byte_size((1,), GGML_TYPE_F32) == 4

    def test_f16_vector(self):
        assert tensor_byte_size((256,), GGML_TYPE_F16) == 512

    def test_f32_matrix(self):
        assert tensor_byte_size((10, 20), GGML_TYPE_F32) == 10 * 20 * 4

    def test_q2k_single_block(self):
        # Q2_K: 256 elements = 1 block = 84 bytes
        assert tensor_byte_size((256,), GGML_TYPE_Q2_K) == 84

    def test_q2k_multiple_blocks(self):
        # 1024 elements = 4 blocks = 4 * 84 = 336 bytes
        assert tensor_byte_size((1024,), GGML_TYPE_Q2_K) == 336

    def test_q4k_calculation(self):
        # Q4_K: 256 elements = 1 block = 144 bytes
        assert tensor_byte_size((256,), GGML_TYPE_Q4_K) == 144

    def test_iq1s_calculation(self):
        # IQ1_S: 256 elements = 1 block = 50 bytes
        assert tensor_byte_size((256,), GGML_TYPE_IQ1_S) == 50

    def test_multidim_shape(self):
        # 256 experts × 1024 rows × 256 cols in Q2_K
        shape = (256, 1024, 256)
        total_elements = 256 * 1024 * 256
        n_blocks = total_elements // QK_K
        expected = n_blocks * 84
        assert tensor_byte_size(shape, GGML_TYPE_Q2_K) == expected

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported GGML type"):
            tensor_byte_size((256,), 999)

    def test_partial_block_rounds_up(self):
        # 257 F16 elements with block_size=1 → 257 blocks → 514 bytes
        assert tensor_byte_size((257,), GGML_TYPE_F16) == 514

    def test_all_known_types_valid(self):
        """All types in GGML_BLOCK_SIZES should compute without error."""
        for dtype in GGML_BLOCK_SIZES:
            result = tensor_byte_size((QK_K,), dtype)
            assert result > 0


class TestParseGGUF:
    """Tests for GGUF file parser edge cases."""

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_gguf("/nonexistent/path/model.gguf")

    def test_corrupt_magic(self, tmp_path):
        """File with wrong magic bytes should raise ValueError."""
        bad_file = tmp_path / "bad.gguf"
        bad_file.write_bytes(b"\x00\x00\x00\x00" + b"\x00" * 100)
        with pytest.raises(ValueError, match="Not a GGUF file"):
            parse_gguf(str(bad_file))

    def test_truncated_file(self, tmp_path):
        """File with valid magic but truncated header should raise."""
        trunc = tmp_path / "truncated.gguf"
        # Write valid magic but nothing else
        trunc.write_bytes(struct.pack('<I', GGUF_MAGIC))
        with pytest.raises(struct.error):
            parse_gguf(str(trunc))

    def test_empty_file(self, tmp_path):
        """Empty file should raise on magic read."""
        empty = tmp_path / "empty.gguf"
        empty.write_bytes(b"")
        with pytest.raises(struct.error):
            parse_gguf(str(empty))

    def test_valid_minimal_gguf(self, tmp_path):
        """Minimal valid GGUF: magic + version + 0 tensors + 0 kv pairs."""
        f = tmp_path / "minimal.gguf"
        data = struct.pack('<I', GGUF_MAGIC)    # magic
        data += struct.pack('<I', 3)             # version
        data += struct.pack('<Q', 0)             # n_tensors
        data += struct.pack('<Q', 0)             # n_kv
        f.write_bytes(data)
        info = parse_gguf(str(f))
        assert info.n_tensors == 0
        assert info.version == 3
        assert info.metadata == {}
        assert info.tensors == {}


class TestParseSplitGGUF:
    """Tests for split GGUF directory parser."""

    def test_no_gguf_files(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .gguf files"):
            parse_split_gguf(str(tmp_path))

    def test_nonexistent_directory(self):
        with pytest.raises(FileNotFoundError):
            parse_split_gguf("/nonexistent/model_dir")


class TestDequantQ2K:
    """Tests for Q2_K dequantization routines."""

    def _make_q2k_block(self):
        """Create a single valid Q2_K block (84 bytes for 256 elements)."""
        scales = np.random.randint(0, 256, size=16, dtype=np.uint8)
        qs = np.random.randint(0, 256, size=64, dtype=np.uint8)
        d = np.float16(0.5)
        dmin = np.float16(0.1)
        block = np.concatenate([
            scales.tobytes(),
            qs.tobytes(),
            np.array([d], dtype=np.float16).tobytes(),
            np.array([dmin], dtype=np.float16).tobytes(),
        ].copy())
        return bytes(block)

    def _make_q2k_block_bytes(self):
        """Create raw bytes for a Q2_K block."""
        scales = bytes(np.random.randint(0, 256, size=16, dtype=np.uint8))
        qs = bytes(np.random.randint(0, 256, size=64, dtype=np.uint8))
        d = np.array([0.5], dtype=np.float16).tobytes()
        dmin = np.array([0.1], dtype=np.float16).tobytes()
        return scales + qs + d + dmin

    def test_scalar_and_fast_agree(self):
        """Scalar and vectorized dequant should produce the same output."""
        block_data = self._make_q2k_block_bytes()
        raw = np.frombuffer(block_data, dtype=np.uint8)
        shape = (256,)
        result_scalar = dequant_q2_k(raw, shape)
        result_fast = dequant_q2_k_fast(raw, shape)
        np.testing.assert_allclose(
            result_scalar.astype(np.float32),
            result_fast.astype(np.float32),
            atol=1e-2,
        )

    def test_output_shape(self):
        """Dequantized output should match the requested shape."""
        block_data = self._make_q2k_block_bytes()
        raw = np.frombuffer(block_data, dtype=np.uint8)
        result = dequant_q2_k_fast(raw, (16, 16))
        assert result.shape == (16, 16)

    def test_output_dtype(self):
        """Dequantized output should be float16."""
        block_data = self._make_q2k_block_bytes()
        raw = np.frombuffer(block_data, dtype=np.uint8)
        result = dequant_q2_k_fast(raw, (256,))
        assert result.dtype == np.float16

    def test_multi_block(self):
        """Multiple Q2_K blocks should dequantize correctly."""
        n_blocks = 4
        blocks = b"".join(self._make_q2k_block_bytes() for _ in range(n_blocks))
        raw = np.frombuffer(blocks, dtype=np.uint8)
        shape = (n_blocks * 256,)
        result = dequant_q2_k_fast(raw, shape)
        assert result.shape == shape
        assert not np.all(result == 0)


class TestGGUFTensorInfo:
    """Tests for GGUFTensorInfo dataclass."""

    def test_basic_fields(self):
        info = GGUFTensorInfo(
            name="blk.0.ffn_gate_exps.weight",
            n_dims=3,
            shape=(256, 1024, 256),
            dtype=GGML_TYPE_Q2_K,
            offset=0,
        )
        assert info.name == "blk.0.ffn_gate_exps.weight"
        assert info.n_dims == 3
        assert info.file_idx == 0
        assert info.total_bytes == 0  # default

    def test_with_total_bytes(self):
        info = GGUFTensorInfo(
            name="test",
            n_dims=1,
            shape=(256,),
            dtype=GGML_TYPE_F16,
            offset=1024,
            file_idx=2,
            total_bytes=512,
        )
        assert info.total_bytes == 512
        assert info.file_idx == 2
        assert info.offset == 1024
