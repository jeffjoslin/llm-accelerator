"""Tests for rotation module."""

import torch
import pytest

from src.rotation.random_rotation import (
    generate_rotation_matrix,
    generate_block_rotation,
    apply_rotation,
    apply_inverse_rotation,
)


class TestGenerateRotationMatrix:
    def test_shape(self):
        Q = generate_rotation_matrix(128)
        assert Q.shape == (128, 128)

    def test_orthogonality(self):
        Q = generate_rotation_matrix(128)
        eye = Q.T @ Q
        assert torch.allclose(eye, torch.eye(128), atol=1e-5)

    def test_orthogonality_small(self):
        Q = generate_rotation_matrix(8)
        eye = Q.T @ Q
        assert torch.allclose(eye, torch.eye(8), atol=1e-6)

    def test_determinism(self):
        Q1 = generate_rotation_matrix(64, seed=42)
        Q2 = generate_rotation_matrix(64, seed=42)
        assert torch.allclose(Q1, Q2)

    def test_different_seeds(self):
        Q1 = generate_rotation_matrix(64, seed=42)
        Q2 = generate_rotation_matrix(64, seed=99)
        assert not torch.allclose(Q1, Q2)

    def test_det_is_pm1(self):
        Q = generate_rotation_matrix(32)
        det = torch.det(Q)
        assert abs(abs(det.item()) - 1.0) < 1e-4


class TestBlockRotation:
    def test_shape(self):
        Q = generate_block_rotation(128, block_size=64)
        assert Q.shape == (128, 128)

    def test_orthogonality(self):
        Q = generate_block_rotation(128, block_size=32)
        eye = Q.T @ Q
        assert torch.allclose(eye, torch.eye(128), atol=1e-5)

    def test_block_diagonal_structure(self):
        Q = generate_block_rotation(8, block_size=4)
        # Off-diagonal blocks should be zero
        assert torch.allclose(Q[:4, 4:], torch.zeros(4, 4), atol=1e-7)
        assert torch.allclose(Q[4:, :4], torch.zeros(4, 4), atol=1e-7)

    def test_invalid_block_size(self):
        with pytest.raises(ValueError):
            generate_block_rotation(10, block_size=3)


class TestApplyRotation:
    def test_shape_preserved(self):
        Q = generate_rotation_matrix(64)
        x = torch.randn(2, 4, 16, 64)
        y = apply_rotation(x, Q)
        assert y.shape == x.shape

    def test_norm_preserved(self):
        Q = generate_rotation_matrix(64)
        x = torch.randn(10, 64)
        y = apply_rotation(x, Q)
        x_norms = x.norm(dim=-1)
        y_norms = y.norm(dim=-1)
        assert torch.allclose(x_norms, y_norms, atol=1e-5)

    def test_round_trip(self):
        Q = generate_rotation_matrix(64)
        x = torch.randn(5, 3, 64)
        y = apply_rotation(x, Q)
        x_back = apply_inverse_rotation(y, Q)
        assert torch.allclose(x, x_back, atol=1e-5)
