"""Quantized Johnson-Lindenstrauss (QJL) residual correction for TurboQuant.

Stage B of TurboQuant: after MSE quantization (Stage A), compute the residual
and apply a 1-bit sign sketch via a random Gaussian projection. This corrects
inner-product bias and yields an unbiased estimator.

Reconstruction formula:
    r̂ = (√(π/2) / d) · γ · S^T · qjl_float

where γ = ‖r‖₂, qjl_float ∈ {-1, +1}^d, and S is i.i.d. N(0,1).
"""

import math

import torch


class QJLSketch:
    """QJL sign sketch for residual correction.

    The projection matrix S is generated once and shared across all
    layers and heads (data-oblivious per paper).
    """

    def __init__(self, dim: int, seed: int = 137):
        """Initialize with random Gaussian projection matrix.

        Args:
            dim: Vector dimension (head_dim).
            seed: Random seed for reproducibility.
        """
        self.dim = dim
        self.seed = seed
        gen = torch.Generator().manual_seed(seed)
        self.S = torch.randn(dim, dim, generator=gen, dtype=torch.float32)

    def sketch(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute sign sketch and residual norm.

        Args:
            r: Residual tensor of shape (..., d).

        Returns:
            signs: Boolean tensor of shape (..., d). True = positive.
            gamma: Residual norm tensor of shape (..., 1).
        """
        gamma = r.norm(dim=-1, keepdim=True)
        S = self.S.to(r.dtype).to(r.device)
        projected = r @ S.T  # (..., d)
        signs = projected >= 0  # bool tensor
        return signs, gamma

    def reconstruct_residual(self, signs: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Reconstruct approximate residual from sign sketch.

        Formula: r̂ = (√(π/2) / d) · γ · S^T · qjl_float

        Args:
            signs: Boolean tensor of shape (..., d).
            gamma: Norm tensor of shape (..., 1).

        Returns:
            Approximate residual of shape (..., d).
        """
        sign_float = signs.float() * 2.0 - 1.0  # {0,1} → {-1,+1}
        S = self.S.to(sign_float.device)
        scale = math.sqrt(math.pi / 2.0) / self.dim
        r_hat = scale * gamma * (sign_float @ S)
        return r_hat

    def to(self, device: torch.device) -> "QJLSketch":
        """Move projection matrix to device."""
        self.S = self.S.to(device)
        return self
