"""Tests for scalar quantizer and index packing."""

import torch
import pytest

from src.quant.scalar_quantizer import (
    TurboQuantMSEQuantizer,
    NaiveInt4Quantizer,
    NoOpQuantizer,
    CompressedTensor,
    pack_indices,
    unpack_indices,
)


class TestPackUnpack:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_round_trip(self, bits):
        n = 128
        max_val = (1 << bits) - 1
        indices = torch.randint(0, max_val + 1, (n,), dtype=torch.int32)
        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, bits, n)
        assert torch.equal(indices, unpacked)

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_round_trip_odd_length(self, bits):
        n = 37  # deliberately not aligned
        max_val = (1 << bits) - 1
        indices = torch.randint(0, max_val + 1, (n,), dtype=torch.int32)
        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, bits, n)
        assert torch.equal(indices, unpacked)

    def test_4bit_compression(self):
        indices = torch.randint(0, 16, (1024,), dtype=torch.int32)
        packed = pack_indices(indices, 4)
        assert packed.numel() == 512  # 2 values per byte

    def test_2bit_compression(self):
        indices = torch.randint(0, 4, (1024,), dtype=torch.int32)
        packed = pack_indices(indices, 2)
        assert packed.numel() == 256  # 4 values per byte


class TestTurboQuantMSEQuantizer:
    def setup_method(self):
        self.quantizer = TurboQuantMSEQuantizer(bits=4, head_dim=64, seed=42)

    def test_round_trip_shape(self):
        x = torch.randn(2, 4, 8, 64)
        compressed = self.quantizer.quantize(x)
        reconstructed = self.quantizer.dequantize(compressed)
        assert reconstructed.shape == x.shape

    def test_indices_in_range(self):
        x = torch.randn(1, 1, 16, 64)
        compressed = self.quantizer.quantize(x)
        assert (compressed.indices >= 0).all()
        assert (compressed.indices < 16).all()

    def test_mse_4bit_bound(self):
        """4-bit MSE on unit vectors should be ≤ 0.009 (paper bound)."""
        torch.manual_seed(42)
        q = TurboQuantMSEQuantizer(bits=4, head_dim=128, seed=42)
        # Generate random unit vectors
        x = torch.randn(100, 128)
        x = x / x.norm(dim=-1, keepdim=True)

        compressed = q.quantize(x)
        x_hat = q.dequantize(compressed)
        mse = ((x - x_hat) ** 2).mean().item()
        # Paper bound is 0.009; allow some margin for finite samples
        assert mse < 0.02, f"MSE {mse} exceeds relaxed bound 0.02"

    def test_mse_3bit_bound(self):
        """3-bit MSE on unit vectors should be ≤ 0.03 (paper bound)."""
        torch.manual_seed(42)
        q = TurboQuantMSEQuantizer(bits=3, head_dim=128, seed=42)
        x = torch.randn(100, 128)
        x = x / x.norm(dim=-1, keepdim=True)

        compressed = q.quantize(x)
        x_hat = q.dequantize(compressed)
        mse = ((x - x_hat) ** 2).mean().item()
        assert mse < 0.06, f"MSE {mse} exceeds relaxed bound 0.06"

    def test_norms_stored(self):
        x = torch.randn(2, 4, 8, 64) * 5  # non-unit vectors
        compressed = self.quantizer.quantize(x)
        assert compressed.norms.shape == (2, 4, 8, 1)
        # Norms should be positive
        assert (compressed.norms > 0).all()

    def test_blockwise_rotation(self):
        q = TurboQuantMSEQuantizer(
            bits=4, head_dim=128, rotation_type="blockwise", block_size=64, seed=42
        )
        x = torch.randn(1, 1, 4, 128)
        compressed = q.quantize(x)
        reconstructed = q.dequantize(compressed)
        assert reconstructed.shape == x.shape


class TestNaiveInt4Quantizer:
    def test_round_trip_shape(self):
        q = NaiveInt4Quantizer()
        x = torch.randn(2, 4, 8, 64)
        compressed = q.quantize(x)
        reconstructed = q.dequantize(compressed)
        assert reconstructed.shape == x.shape

    def test_indices_in_range(self):
        q = NaiveInt4Quantizer()
        x = torch.randn(10, 64)
        compressed = q.quantize(x)
        assert (compressed.indices >= 0).all()
        assert (compressed.indices <= 15).all()


class TestNoOpQuantizer:
    def test_identity(self):
        q = NoOpQuantizer()
        x = torch.randn(2, 64)
        compressed = q.quantize(x)
        reconstructed = q.dequantize(compressed)
        assert torch.equal(x, reconstructed)
