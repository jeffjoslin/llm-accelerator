"""Tests for codebook module."""

import math

import torch
import pytest

from src.quant.codebook import compute_codebook, nearest_codebook_index, codebook_lookup


class TestComputeCodebook:
    def test_1bit_values(self):
        cb = compute_codebook(1, 128)
        expected_val = math.sqrt(2.0 / (math.pi * 128))
        assert len(cb) == 2
        assert abs(cb[0].item() + expected_val) < 1e-6
        assert abs(cb[1].item() - expected_val) < 1e-6

    def test_2bit_values(self):
        cb = compute_codebook(2, 128)
        scale = 1.0 / math.sqrt(128)
        assert len(cb) == 4
        assert abs(cb[0].item() - (-1.51 * scale)) < 1e-5
        assert abs(cb[1].item() - (-0.453 * scale)) < 1e-5
        assert abs(cb[2].item() - (0.453 * scale)) < 1e-5
        assert abs(cb[3].item() - (1.51 * scale)) < 1e-5

    def test_3bit_count(self):
        cb = compute_codebook(3, 128)
        assert len(cb) == 8

    def test_4bit_count(self):
        cb = compute_codebook(4, 128)
        assert len(cb) == 16

    def test_codebook_sorted(self):
        for bits in [1, 2, 3, 4]:
            cb = compute_codebook(bits, 128)
            for i in range(len(cb) - 1):
                assert cb[i] < cb[i + 1], f"Codebook not sorted for bits={bits}"

    def test_codebook_symmetric(self):
        """Codebook should be approximately symmetric around 0."""
        for bits in [1, 2, 3, 4]:
            cb = compute_codebook(bits, 128)
            # Sum should be close to 0 for symmetric distribution
            assert abs(cb.sum().item()) < 0.01

    def test_codebook_reasonable_range(self):
        """Codebook entries should be within a reasonable range around 1/√d.

        Note: the paper's closed-form b=2 values (±1.51/√d) extend beyond
        the strict [-1/√d, +1/√d] support of the Beta distribution. This is
        expected — outer centroids represent the tails and can exceed the
        support while still being optimal for MSE.
        """
        d = 128
        bound = 5.0 / math.sqrt(d)  # generous bound: 5σ of the coordinate distribution
        for bits in [1, 2, 3, 4]:
            cb = compute_codebook(bits, d)
            assert (cb >= -bound).all(), f"bits={bits}: {cb.min()} < {-bound}"
            assert (cb <= bound).all(), f"bits={bits}: {cb.max()} > {bound}"

    def test_caching(self):
        """Same args should return same object (lru_cache)."""
        cb1 = compute_codebook(4, 128)
        cb2 = compute_codebook(4, 128)
        assert cb1 is cb2


class TestNearestCodebookIndex:
    def test_basic(self):
        cb = torch.tensor([-1.0, 0.0, 1.0])
        y = torch.tensor([[0.6, -0.8, 0.1]])
        idx = nearest_codebook_index(y, cb)
        assert idx[0, 0].item() == 2  # 0.6 → 1.0
        assert idx[0, 1].item() == 0  # -0.8 → -1.0
        assert idx[0, 2].item() == 1  # 0.1 → 0.0

    def test_shape_preserved(self):
        cb = compute_codebook(4, 128)
        y = torch.randn(2, 4, 8, 128) * 0.01
        idx = nearest_codebook_index(y, cb)
        assert idx.shape == (2, 4, 8, 128)

    def test_index_range(self):
        for bits in [1, 2, 3, 4]:
            cb = compute_codebook(bits, 64)
            y = torch.randn(100, 64) * 0.1
            idx = nearest_codebook_index(y, cb)
            assert (idx >= 0).all()
            assert (idx < (1 << bits)).all()


class TestCodebookLookup:
    def test_round_trip(self):
        cb = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        idx = torch.tensor([[0, 2, 3, 1]])
        vals = codebook_lookup(idx, cb)
        expected = torch.tensor([[-1.0, 0.5, 1.0, 0.0]])
        assert torch.allclose(vals, expected)
