"""Scalar codebook generation for TurboQuant MSE quantizer.

After random rotation, each coordinate of a unit-norm d-dimensional vector
follows a Beta(1/2, (d-1)/2) distribution rescaled to [-1/√d, +1/√d].

We compute Lloyd-Max optimal scalar codebooks for this distribution.
For b=1 and b=2, closed-form solutions from the paper are used.
For b>=3, we run Lloyd-Max iterations numerically.
"""

import math
from functools import lru_cache

import numpy as np
import torch


def _beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """PDF of a single coordinate after random rotation of a unit vector.

    From the paper (Lemma 1): if x ∈ S^{d-1} is uniform on the unit sphere,
    each coordinate follows:
        f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)
    for x ∈ [-1, 1].

    Args:
        x: Array of coordinate values in [-1, 1].
        d: Vector dimension.

    Returns:
        Unnormalized density values.
    """
    pdf = np.zeros_like(x)
    mask = np.abs(x) < 1.0
    inner = 1.0 - x[mask] ** 2
    # Clamp for numerical stability
    inner = np.maximum(inner, 0.0)
    exponent = (d - 3) / 2.0
    if exponent >= 0:
        pdf[mask] = inner**exponent
    else:
        # d=2 case: exponent = -0.5
        pdf[mask] = inner**exponent
    return pdf


def _lloyd_max(d: int, num_levels: int, max_iter: int = 500, tol: float = 1e-10) -> np.ndarray:
    """Compute Lloyd-Max optimal codebook for the coordinate distribution.

    Args:
        d: Vector dimension.
        num_levels: Number of quantization levels (2^b).
        max_iter: Maximum iterations.
        tol: Convergence tolerance on codebook change.

    Returns:
        Sorted array of codebook centroids, shape (num_levels,).
    """
    # The support is [-1, 1] but mass concentrates near 0 for large d.
    # Use a practical bound that captures >99.99% of the mass.
    # For the Beta distribution on the sphere, std ≈ 1/√d.
    # Use ±min(1, 5/√d) to capture the tails without wasting grid resolution.
    practical_bound = min(1.0, 5.0 / math.sqrt(d))
    # Fine grid for numerical integration
    num_points = 100_000
    z = np.linspace(-practical_bound, practical_bound, num_points)
    dz = z[1] - z[0]
    pdf = _beta_pdf(z, d)
    pdf = pdf / (pdf.sum() * dz)  # normalize

    # Initialize centroids uniformly within the practical range
    centroids = np.linspace(-practical_bound * 0.9, practical_bound * 0.9, num_levels)

    for _ in range(max_iter):
        # Assign each grid point to nearest centroid
        dists = np.abs(z[:, None] - centroids[None, :])  # (num_points, num_levels)
        assignments = np.argmin(dists, axis=1)

        # Update centroids as weighted mean within each partition
        new_centroids = np.zeros(num_levels)
        for k in range(num_levels):
            mask = assignments == k
            weight = pdf[mask] * dz
            total_weight = weight.sum()
            if total_weight > 0:
                new_centroids[k] = (z[mask] * weight).sum() / total_weight
            else:
                new_centroids[k] = centroids[k]

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    return np.sort(centroids)


@lru_cache(maxsize=32)
def compute_codebook(bits: int, dim: int) -> torch.Tensor:
    """Compute or return cached optimal scalar codebook.

    Args:
        bits: Bit-width (1, 2, 3, or 4).
        dim: Vector dimension.

    Returns:
        Tensor of shape (2^bits,) with sorted codebook centroids.
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be 1-8, got {bits}")

    num_levels = 1 << bits

    # Clear LRU cache since codebook computation has changed
    if bits == 1:
        # Closed-form from paper: ±√(2/(πd))
        # These are on the [-1, 1] scale (coordinate of unit vector on sphere)
        val = math.sqrt(2.0 / (math.pi * dim))
        centroids = np.array([-val, val])
    elif bits == 2:
        # Closed-form from paper: ±0.453/√d, ±1.51/√d
        scale = 1.0 / math.sqrt(dim)
        centroids = np.array([-1.51 * scale, -0.453 * scale, 0.453 * scale, 1.51 * scale])
    else:
        # Lloyd-Max numerical optimization
        centroids = _lloyd_max(dim, num_levels)

    return torch.from_numpy(centroids).float()


def nearest_codebook_index(y: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Find nearest codebook entry for each element.

    Args:
        y: Tensor of shape (..., d) with rotated coordinates.
        codebook: Tensor of shape (num_levels,) with sorted centroids.

    Returns:
        Index tensor of same shape as y, dtype torch.int32.
    """
    # Expand codebook for broadcasting: (1, ..., 1, num_levels)
    cb = codebook.to(y.device).to(y.dtype)
    # Compute distances: (..., d, num_levels)
    dists = torch.abs(y.unsqueeze(-1) - cb)
    return dists.argmin(dim=-1).to(torch.int32)


def codebook_lookup(indices: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Look up codebook values from indices.

    Args:
        indices: Integer tensor of shape (..., d).
        codebook: Tensor of shape (num_levels,).

    Returns:
        Tensor of shape (..., d) with codebook values.
    """
    return codebook.to(device=indices.device, dtype=torch.float32)[indices.long()]
