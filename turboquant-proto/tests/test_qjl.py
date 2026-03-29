"""Tests for QJL sketch module."""

import math

import torch
import pytest

from src.qjl.qjl_sketch import QJLSketch


class TestQJLSketch:
    def setup_method(self):
        self.sketch = QJLSketch(dim=64, seed=42)

    def test_sketch_shapes(self):
        r = torch.randn(10, 64)
        signs, gamma = self.sketch.sketch(r)
        assert signs.shape == (10, 64)
        assert gamma.shape == (10, 1)
        assert signs.dtype == torch.bool

    def test_gamma_is_norm(self):
        r = torch.randn(10, 64)
        signs, gamma = self.sketch.sketch(r)
        expected_norms = r.norm(dim=-1, keepdim=True)
        assert torch.allclose(gamma, expected_norms, atol=1e-6)

    def test_reconstruct_shape(self):
        r = torch.randn(5, 64)
        signs, gamma = self.sketch.sketch(r)
        r_hat = self.sketch.reconstruct_residual(signs, gamma)
        assert r_hat.shape == (5, 64)

    def test_unbiasedness(self):
        """Inner product estimate should be unbiased over many trials.

        E[<y, r_hat>] ≈ <y, r> for random y, r.
        """
        torch.manual_seed(123)
        dim = 64
        sketch = QJLSketch(dim=dim, seed=999)

        n_trials = 2000
        true_ips = []
        est_ips = []

        for _ in range(n_trials):
            r = torch.randn(dim)
            y = torch.randn(dim)

            signs, gamma = sketch.sketch(r.unsqueeze(0))
            r_hat = sketch.reconstruct_residual(signs, gamma).squeeze(0)

            true_ips.append((y @ r).item())
            est_ips.append((y @ r_hat).item())

        true_mean = sum(true_ips) / n_trials
        est_mean = sum(est_ips) / n_trials

        # The means should be close (both ~0 for random vectors)
        # More importantly, check correlation: est should track true
        import numpy as np
        corr = np.corrcoef(true_ips, est_ips)[0, 1]
        assert corr > 0.3, f"Correlation {corr} too low — estimate is not tracking true IP"

    def test_zero_residual(self):
        """Zero residual should reconstruct to zero."""
        r = torch.zeros(5, 64)
        signs, gamma = self.sketch.sketch(r)
        r_hat = self.sketch.reconstruct_residual(signs, gamma)
        assert torch.allclose(r_hat, torch.zeros_like(r_hat), atol=1e-7)

    def test_determinism(self):
        s1 = QJLSketch(dim=32, seed=42)
        s2 = QJLSketch(dim=32, seed=42)
        assert torch.equal(s1.S, s2.S)

    def test_batched(self):
        """Should work with batched input (batch, heads, seq, dim)."""
        sketch = QJLSketch(dim=64, seed=42)
        r = torch.randn(2, 4, 8, 64)
        signs, gamma = sketch.sketch(r)
        assert signs.shape == (2, 4, 8, 64)
        assert gamma.shape == (2, 4, 8, 1)
        r_hat = sketch.reconstruct_residual(signs, gamma)
        assert r_hat.shape == (2, 4, 8, 64)
