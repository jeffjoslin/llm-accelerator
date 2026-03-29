"""TurboQuant scalar quantizer — MSE-optimal Stage A.

Pipeline:
1. Normalize input vectors to unit norm (store norms as metadata).
2. Apply random rotation Π.
3. Quantize each coordinate to nearest codebook entry.
4. Store packed indices + norms.

Dequantization:
1. Unpack indices → codebook lookup.
2. Apply inverse rotation Π^T.
3. Rescale by stored norms.
"""

from dataclasses import dataclass

import torch

from src.quant.codebook import codebook_lookup, compute_codebook, nearest_codebook_index
from src.rotation.random_rotation import (
    apply_inverse_rotation,
    apply_rotation,
    generate_block_rotation,
    generate_rotation_matrix,
)


@dataclass
class CompressedTensor:
    """Compressed representation of a tensor after MSE quantization."""

    indices: torch.Tensor  # int32, shape (..., d) — codebook indices
    norms: torch.Tensor  # float32, shape (..., 1) — per-vector norms
    bits: int  # bit-width used
    codebook: torch.Tensor  # shape (2^bits,) — the codebook used


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack low-bit indices into uint8 tensor for memory savings.

    Args:
        indices: int32 tensor with values in [0, 2^bits).
        bits: Bit-width (1, 2, 3, or 4).

    Returns:
        Packed uint8 tensor. For bits=4, 2 values per byte.
        For bits=2, 4 values per byte. For bits=1, 8 values per byte.
        For bits=3, packs 8 values into 3 bytes.
    """
    flat = indices.reshape(-1).to(torch.uint8)
    n = flat.numel()

    if bits == 4:
        # 2 values per byte
        pad = (2 - n % 2) % 2
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)])
        packed = (flat[0::2] << 4) | flat[1::2]
        return packed

    if bits == 2:
        # 4 values per byte
        pad = (4 - n % 4) % 4
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)])
        packed = (flat[0::4] << 6) | (flat[1::4] << 4) | (flat[2::4] << 2) | flat[3::4]
        return packed

    if bits == 1:
        # 8 values per byte
        pad = (8 - n % 8) % 8
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)])
        packed = torch.zeros(len(flat) // 8, dtype=torch.uint8)
        for i in range(8):
            packed |= flat[i::8] << (7 - i)
        return packed

    if bits == 3:
        # 8 values → 3 bytes (24 bits)
        pad = (8 - n % 8) % 8
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)])
        n_groups = len(flat) // 8
        flat = flat.reshape(n_groups, 8)
        # Pack 8 x 3-bit values into 3 bytes
        b0 = (flat[:, 0] << 5) | (flat[:, 1] << 2) | (flat[:, 2] >> 1)
        b1 = ((flat[:, 2] & 1) << 7) | (flat[:, 3] << 4) | (flat[:, 4] << 1) | (flat[:, 5] >> 2)
        b2 = ((flat[:, 5] & 0x3) << 6) | (flat[:, 6] << 3) | flat[:, 7]
        packed = torch.stack([b0, b1, b2], dim=1).reshape(-1)
        return packed

    raise ValueError(f"Unsupported bits={bits}")


def unpack_indices(packed: torch.Tensor, bits: int, num_elements: int) -> torch.Tensor:
    """Unpack uint8 tensor back to index tensor.

    Args:
        packed: Packed uint8 tensor from pack_indices.
        bits: Bit-width used for packing.
        num_elements: Original number of elements.

    Returns:
        int32 tensor of shape (num_elements,).
    """
    mask = (1 << bits) - 1

    if bits == 4:
        high = (packed >> 4) & mask
        low = packed & mask
        flat = torch.stack([high, low], dim=1).reshape(-1)
        return flat[:num_elements].to(torch.int32)

    if bits == 2:
        v0 = (packed >> 6) & mask
        v1 = (packed >> 4) & mask
        v2 = (packed >> 2) & mask
        v3 = packed & mask
        flat = torch.stack([v0, v1, v2, v3], dim=1).reshape(-1)
        return flat[:num_elements].to(torch.int32)

    if bits == 1:
        parts = []
        for i in range(8):
            parts.append((packed >> (7 - i)) & 1)
        flat = torch.stack(parts, dim=1).reshape(-1)
        return flat[:num_elements].to(torch.int32)

    if bits == 3:
        packed = packed.reshape(-1, 3)
        b0, b1, b2 = packed[:, 0], packed[:, 1], packed[:, 2]
        v0 = (b0 >> 5) & 0x7
        v1 = (b0 >> 2) & 0x7
        v2 = ((b0 & 0x3) << 1) | ((b1 >> 7) & 0x1)
        v3 = (b1 >> 4) & 0x7
        v4 = (b1 >> 1) & 0x7
        v5 = ((b1 & 0x1) << 2) | ((b2 >> 6) & 0x3)
        v6 = (b2 >> 3) & 0x7
        v7 = b2 & 0x7
        flat = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=1).reshape(-1)
        return flat[:num_elements].to(torch.int32)

    raise ValueError(f"Unsupported bits={bits}")


class TurboQuantMSEQuantizer:
    """TurboQuant Stage A: rotation + scalar codebook quantization."""

    def __init__(
        self,
        bits: int,
        head_dim: int,
        rotation_type: str = "full",
        block_size: int = 64,
        seed: int = 42,
    ):
        self.bits = bits
        self.head_dim = head_dim
        self.rotation_type = rotation_type
        self.seed = seed

        # Precompute rotation matrix
        if rotation_type == "blockwise":
            self.rotation = generate_block_rotation(head_dim, block_size, seed=seed)
        else:
            self.rotation = generate_rotation_matrix(head_dim, seed=seed)

        # Precompute codebook
        self.codebook = compute_codebook(bits, head_dim)

    def quantize(self, x: torch.Tensor) -> CompressedTensor:
        """Quantize tensor using rotation + scalar codebook.

        Args:
            x: Tensor of shape (..., head_dim). Typically (batch, heads, seq, head_dim).

        Returns:
            CompressedTensor with indices, norms, and metadata.
        """
        # Store norms for rescaling
        norms = x.norm(dim=-1, keepdim=True)
        # Normalize to unit norm
        x_unit = x / (norms + 1e-8)
        # Rotate
        y = apply_rotation(x_unit, self.rotation)
        # Quantize each coordinate
        indices = nearest_codebook_index(y, self.codebook)
        return CompressedTensor(
            indices=indices,
            norms=norms,
            bits=self.bits,
            codebook=self.codebook,
        )

    def dequantize(self, compressed: CompressedTensor) -> torch.Tensor:
        """Dequantize compressed tensor.

        Args:
            compressed: CompressedTensor from quantize().

        Returns:
            Reconstructed tensor of original shape.
        """
        # Lookup codebook values
        y_hat = codebook_lookup(compressed.indices, compressed.codebook)
        y_hat = y_hat.to(compressed.norms.device)
        # Inverse rotate
        x_hat = apply_inverse_rotation(y_hat, self.rotation)
        # Rescale
        return x_hat * compressed.norms

    def to(self, device: torch.device) -> "TurboQuantMSEQuantizer":
        """Move precomputed tensors to device."""
        self.rotation = self.rotation.to(device)
        self.codebook = self.codebook.to(device)
        return self


class NaiveInt4Quantizer:
    """Simple per-tensor min/max 4-bit quantizer for comparison baseline."""

    def quantize(self, x: torch.Tensor) -> CompressedTensor:
        vmin = x.min()
        vmax = x.max()
        scale = (vmax - vmin) / 15.0
        zero_point = vmin
        indices = ((x - zero_point) / (scale + 1e-8)).round().clamp(0, 15).to(torch.int32)
        # Store scale and zero_point in norms field (repurposed)
        meta = torch.tensor([scale.item(), zero_point.item()])
        return CompressedTensor(indices=indices, norms=meta, bits=4, codebook=torch.empty(0))

    def dequantize(self, compressed: CompressedTensor) -> torch.Tensor:
        scale = compressed.norms[0].item()
        zero_point = compressed.norms[1].item()
        return compressed.indices.float() * scale + zero_point

    def to(self, device: torch.device) -> "NaiveInt4Quantizer":
        return self


class NoOpQuantizer:
    """Pass-through quantizer for baseline validation."""

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def dequantize(self, compressed: torch.Tensor) -> torch.Tensor:
        return compressed

    def to(self, device: torch.device) -> "NoOpQuantizer":
        return self
