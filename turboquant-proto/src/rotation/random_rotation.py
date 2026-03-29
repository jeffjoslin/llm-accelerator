"""Random orthogonal rotation matrices for TurboQuant.

The paper requires randomly rotating input vectors before scalar quantization.
This makes coordinates approximately independent and Beta-distributed,
enabling optimal scalar codebooks.

Generation method: QR decomposition of a random Gaussian matrix produces
a uniformly random orthogonal matrix (Haar measure on O(d)).
"""

import numpy as np
import torch


def generate_rotation_matrix(d: int, seed: int = 42, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate a d×d random orthogonal rotation matrix.

    Uses QR decomposition of a matrix with i.i.d. N(0,1) entries.
    Sign of diagonal of R is used to fix the sign ambiguity so
    the result is uniformly distributed on O(d).

    Args:
        d: Dimension (typically head_dim, e.g. 128).
        seed: Random seed for reproducibility.
        dtype: Output dtype.

    Returns:
        Orthogonal matrix Q of shape (d, d).
    """
    rng = np.random.RandomState(seed)
    G = rng.randn(d, d).astype(np.float64)
    Q, R = np.linalg.qr(G)
    # Fix sign ambiguity: ensure diag(R) > 0
    sign = np.sign(np.diag(R))
    sign[sign == 0] = 1.0
    Q = Q * sign[np.newaxis, :]
    return torch.from_numpy(Q).to(dtype)


def generate_block_rotation(
    d: int, block_size: int, seed: int = 42, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Generate a block-diagonal orthogonal rotation matrix.

    Instead of a full d×d rotation, uses independent block_size×block_size
    rotations along the diagonal. This is faster to apply (O(d·block_size)
    instead of O(d²)) but provides weaker statistical guarantees.

    Args:
        d: Full dimension. Must be divisible by block_size.
        block_size: Size of each diagonal block.
        seed: Random seed.
        dtype: Output dtype.

    Returns:
        Block-diagonal orthogonal matrix of shape (d, d).
    """
    if d % block_size != 0:
        raise ValueError(f"d={d} must be divisible by block_size={block_size}")

    num_blocks = d // block_size
    blocks = []
    for i in range(num_blocks):
        block = generate_rotation_matrix(block_size, seed=seed + i, dtype=dtype)
        blocks.append(block)

    return torch.block_diag(*blocks)


def apply_rotation(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply rotation to the last dimension of x.

    Args:
        x: Tensor of shape (..., d).
        rotation: Orthogonal matrix of shape (d, d).

    Returns:
        Rotated tensor of same shape as x.
    """
    return x @ rotation.to(x.dtype).to(x.device).T


def apply_inverse_rotation(y: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply inverse (transpose) rotation to the last dimension of y.

    For orthogonal Q, Q^{-1} = Q^T.

    Args:
        y: Tensor of shape (..., d).
        rotation: Orthogonal matrix of shape (d, d).

    Returns:
        Inverse-rotated tensor of same shape as y.
    """
    return y @ rotation.to(y.dtype).to(y.device)
