"""Tests for KV-cache interception layer."""

import torch
import pytest

from src.cache_hooks.quantized_cache import (
    TurboQuantCache,
    TurboQuantConfig,
    TurboQuantFullQuantizer,
    build_quantizer,
)
from src.quant.scalar_quantizer import (
    NoOpQuantizer,
    NaiveInt4Quantizer,
    TurboQuantMSEQuantizer,
)


class TestBuildQuantizer:
    def test_baseline(self):
        config = TurboQuantConfig(mode="baseline")
        q = build_quantizer(config)
        assert isinstance(q, NoOpQuantizer)

    def test_int4_naive(self):
        config = TurboQuantConfig(mode="int4_naive")
        q = build_quantizer(config)
        assert isinstance(q, NaiveInt4Quantizer)

    def test_turboquant_mse(self):
        config = TurboQuantConfig(mode="turboquant_mse", bits=4, head_dim=64)
        q = build_quantizer(config)
        assert isinstance(q, TurboQuantMSEQuantizer)

    def test_turboquant_full(self):
        config = TurboQuantConfig(mode="turboquant_full", bits=4, head_dim=64)
        q = build_quantizer(config)
        assert isinstance(q, TurboQuantFullQuantizer)

    def test_unknown_mode(self):
        config = TurboQuantConfig(mode="unknown")
        with pytest.raises(ValueError):
            build_quantizer(config)


class TestTurboQuantCache:
    def _make_cache(self, mode="baseline", bits=4, head_dim=64):
        config = TurboQuantConfig(mode=mode, bits=bits, head_dim=head_dim)
        return TurboQuantCache(config)

    def test_baseline_update(self):
        cache = self._make_cache("baseline")
        k = torch.randn(1, 4, 1, 64)
        v = torch.randn(1, 4, 1, 64)
        all_k, all_v = cache.update(k, v, layer_idx=0)
        assert all_k.shape == (1, 4, 1, 64)
        assert all_v.shape == (1, 4, 1, 64)

    def test_baseline_identity(self):
        """Baseline mode should return exact same values."""
        cache = self._make_cache("baseline")
        k = torch.randn(1, 4, 1, 64)
        v = torch.randn(1, 4, 1, 64)
        all_k, all_v = cache.update(k, v, layer_idx=0)
        assert torch.equal(all_k, k)
        assert torch.equal(all_v, v)

    def test_sequence_concatenation(self):
        """Multiple updates should concatenate along seq dim."""
        cache = self._make_cache("baseline")
        k1 = torch.randn(1, 4, 3, 64)
        v1 = torch.randn(1, 4, 3, 64)
        cache.update(k1, v1, layer_idx=0)

        k2 = torch.randn(1, 4, 1, 64)
        v2 = torch.randn(1, 4, 1, 64)
        all_k, all_v = cache.update(k2, v2, layer_idx=0)
        assert all_k.shape == (1, 4, 4, 64)
        assert all_v.shape == (1, 4, 4, 64)

    def test_seq_length_tracking(self):
        cache = self._make_cache("baseline")
        assert cache.get_seq_length(0) == 0

        k = torch.randn(1, 4, 5, 64)
        v = torch.randn(1, 4, 5, 64)
        cache.update(k, v, layer_idx=0)
        assert cache.get_seq_length(0) == 5

        k2 = torch.randn(1, 4, 1, 64)
        v2 = torch.randn(1, 4, 1, 64)
        cache.update(k2, v2, layer_idx=0)
        assert cache.get_seq_length(0) == 6

    def test_multiple_layers(self):
        cache = self._make_cache("baseline")
        for layer in range(3):
            k = torch.randn(1, 4, 2, 64)
            v = torch.randn(1, 4, 2, 64)
            cache.update(k, v, layer_idx=layer)
        assert len(cache) == 3

    def test_getitem(self):
        cache = self._make_cache("baseline")
        k = torch.randn(1, 4, 3, 64)
        v = torch.randn(1, 4, 3, 64)
        cache.update(k, v, layer_idx=0)
        keys, values = cache[0]
        assert keys.shape == (1, 4, 3, 64)

    def test_mse_mode_runs(self):
        """TurboQuant MSE mode should run without errors."""
        cache = self._make_cache("turboquant_mse", bits=4, head_dim=64)
        k = torch.randn(1, 4, 1, 64)
        v = torch.randn(1, 4, 1, 64)
        all_k, all_v = cache.update(k, v, layer_idx=0)
        assert all_k.shape == (1, 4, 1, 64)
        assert all_v.shape == (1, 4, 1, 64)

    def test_full_mode_runs(self):
        """TurboQuant full mode should run without errors."""
        cache = self._make_cache("turboquant_full", bits=4, head_dim=64)
        k = torch.randn(1, 4, 1, 64)
        v = torch.randn(1, 4, 1, 64)
        all_k, all_v = cache.update(k, v, layer_idx=0)
        assert all_k.shape == (1, 4, 1, 64)
        assert all_v.shape == (1, 4, 1, 64)

    def test_int4_naive_mode_runs(self):
        cache = self._make_cache("int4_naive")
        k = torch.randn(1, 4, 1, 64)
        v = torch.randn(1, 4, 1, 64)
        all_k, all_v = cache.update(k, v, layer_idx=0)
        assert all_k.shape == (1, 4, 1, 64)

    def test_mse_multi_step(self):
        """MSE mode should handle multiple generation steps."""
        cache = self._make_cache("turboquant_mse", bits=4, head_dim=64)
        # Prefill
        k1 = torch.randn(1, 4, 8, 64)
        v1 = torch.randn(1, 4, 8, 64)
        cache.update(k1, v1, layer_idx=0)

        # Decode steps
        for _ in range(5):
            k = torch.randn(1, 4, 1, 64)
            v = torch.randn(1, 4, 1, 64)
            all_k, all_v = cache.update(k, v, layer_idx=0)

        assert all_k.shape == (1, 4, 13, 64)
        assert cache.get_seq_length(0) == 13


class TestTurboQuantFullQuantizer:
    def test_quantize_dequantize_shape(self):
        q = TurboQuantFullQuantizer(bits=4, head_dim=64, seed=42)
        x = torch.randn(2, 4, 8, 64)
        compressed = q.quantize(x)
        reconstructed = q.dequantize(compressed)
        assert reconstructed.shape == x.shape

    def test_min_bits(self):
        with pytest.raises(ValueError):
            TurboQuantFullQuantizer(bits=1, head_dim=64)

    def test_mse_uses_fewer_bits(self):
        """Full quantizer should use (bits-1) for MSE stage."""
        q = TurboQuantFullQuantizer(bits=4, head_dim=64)
        assert q.mse_quantizer.bits == 3
