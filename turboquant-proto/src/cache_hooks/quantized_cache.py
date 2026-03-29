"""KV-cache interception layer for TurboQuant.

Subclasses HuggingFace DynamicCache to intercept K/V tensors on write,
quantize them, and dequantize on read for attention.

Modes:
  - baseline: NoOp pass-through (for validation)
  - int4_naive: Simple per-tensor 4-bit quantization
  - turboquant_mse: TurboQuant Stage A (rotation + scalar codebook)
  - turboquant_full: TurboQuant Stage A + B (MSE + QJL residual)
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import yaml

from src.qjl.qjl_sketch import QJLSketch
from src.quant.scalar_quantizer import (
    CompressedTensor,
    NaiveInt4Quantizer,
    NoOpQuantizer,
    TurboQuantMSEQuantizer,
)


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant cache."""

    mode: str = "baseline"
    bits: int = 4
    quantize_keys: bool = True
    quantize_values: bool = True
    rotation_enabled: bool = True
    rotation_type: str = "full"
    block_size: int = 64
    qjl_enabled: bool = False
    seed: int = 42
    head_dim: int = 128

    @classmethod
    def from_yaml(cls, path: str) -> "TurboQuantConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: dict) -> "TurboQuantConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class FullCompressed:
    """Compressed representation for TurboQuant full (MSE + QJL)."""

    __slots__ = ("mse", "signs", "gamma")

    def __init__(self, mse: CompressedTensor, signs: torch.Tensor, gamma: torch.Tensor):
        self.mse = mse
        self.signs = signs
        self.gamma = gamma


class TurboQuantFullQuantizer:
    """TurboQuant Stage A + B: MSE quantization + QJL residual correction.

    Total bit budget = bits. Uses (bits-1) for MSE stage and 1 bit for QJL.
    """

    def __init__(
        self,
        bits: int,
        head_dim: int,
        rotation_type: str = "full",
        block_size: int = 64,
        seed: int = 42,
    ):
        if bits < 2:
            raise ValueError("Full TurboQuant requires at least 2 bits (1 for MSE + 1 for QJL)")
        self.bits = bits
        self.mse_quantizer = TurboQuantMSEQuantizer(
            bits=bits - 1,
            head_dim=head_dim,
            rotation_type=rotation_type,
            block_size=block_size,
            seed=seed,
        )
        self.qjl = QJLSketch(dim=head_dim, seed=seed + 1000)

    def quantize(self, x: torch.Tensor) -> FullCompressed:
        mse_compressed = self.mse_quantizer.quantize(x)
        x_hat_mse = self.mse_quantizer.dequantize(mse_compressed)
        residual = x - x_hat_mse
        signs, gamma = self.qjl.sketch(residual)
        return FullCompressed(mse=mse_compressed, signs=signs, gamma=gamma)

    def dequantize(self, compressed: FullCompressed) -> torch.Tensor:
        x_hat_mse = self.mse_quantizer.dequantize(compressed.mse)
        r_hat = self.qjl.reconstruct_residual(compressed.signs, compressed.gamma)
        return x_hat_mse + r_hat

    def to(self, device: torch.device) -> "TurboQuantFullQuantizer":
        self.mse_quantizer.to(device)
        self.qjl.to(device)
        return self


def build_quantizer(config: TurboQuantConfig):
    """Factory: build the appropriate quantizer from config."""
    if config.mode == "baseline":
        return NoOpQuantizer()
    elif config.mode == "int4_naive":
        return NaiveInt4Quantizer()
    elif config.mode == "turboquant_mse":
        return TurboQuantMSEQuantizer(
            bits=config.bits,
            head_dim=config.head_dim,
            rotation_type=config.rotation_type,
            block_size=config.block_size,
            seed=config.seed,
        )
    elif config.mode == "turboquant_full":
        return TurboQuantFullQuantizer(
            bits=config.bits,
            head_dim=config.head_dim,
            rotation_type=config.rotation_type,
            block_size=config.block_size,
            seed=config.seed,
        )
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


class TurboQuantCache:
    """KV cache with optional quantization.

    Drop-in replacement for HuggingFace DynamicCache. Stores compressed
    K/V tensors and dequantizes on read.

    Uses the same interface as transformers.DynamicCache:
      - update(key_states, value_states, layer_idx, cache_kwargs)
      - __getitem__(layer_idx) → (keys, values)
      - get_seq_length(layer_idx)
    """

    def __init__(self, config: TurboQuantConfig):
        self.config = config
        self.key_quantizer = build_quantizer(config) if config.quantize_keys else NoOpQuantizer()
        self.value_quantizer = build_quantizer(config) if config.quantize_values else NoOpQuantizer()
        self._compressed_keys: list[list] = []
        self._compressed_values: list[list] = []
        self._seen_tokens = 0
        # Track key/value shapes for seq_length computation
        self._seq_lengths: list[int] = []
        self._dtype: Optional[torch.dtype] = None  # track input dtype for dequant

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store new KV states (quantized) and return full (dequantized) cache.

        Args:
            key_states: (batch, num_heads, seq_len, head_dim)
            value_states: (batch, num_heads, seq_len, head_dim)
            layer_idx: Which transformer layer.
            cache_kwargs: Additional kwargs (unused for now).

        Returns:
            (all_keys, all_values) — dequantized, concatenated along seq dim.
        """
        # Extend storage if needed
        while len(self._compressed_keys) <= layer_idx:
            self._compressed_keys.append([])
            self._compressed_values.append([])
            self._seq_lengths.append(0)

        # Track input dtype for casting dequantized output
        if self._dtype is None:
            self._dtype = key_states.dtype

        # Track tokens (only count from layer 0)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]

        # Quantize and store
        compressed_k = self.key_quantizer.quantize(key_states)
        compressed_v = self.value_quantizer.quantize(value_states)
        self._compressed_keys[layer_idx].append(compressed_k)
        self._compressed_values[layer_idx].append(compressed_v)
        self._seq_lengths[layer_idx] += key_states.shape[2]

        # Dequantize full sequence for attention
        all_keys = self._dequantize_all(self._compressed_keys[layer_idx], self.key_quantizer)
        all_values = self._dequantize_all(self._compressed_values[layer_idx], self.value_quantizer)
        return all_keys, all_values

    def _dequantize_all(self, compressed_list: list, quantizer) -> torch.Tensor:
        """Dequantize and concatenate all chunks for a layer."""
        chunks = [quantizer.dequantize(c) for c in compressed_list]
        result = torch.cat(chunks, dim=2)  # cat along seq dim
        # Cast back to original dtype (quantization works in float32 internally)
        if self._dtype is not None and result.dtype != self._dtype:
            result = result.to(self._dtype)
        return result

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._seq_lengths):
            return 0
        return self._seq_lengths[layer_idx]

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def __len__(self) -> int:
        return len(self._compressed_keys)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self._dequantize_all(self._compressed_keys[layer_idx], self.key_quantizer)
        values = self._dequantize_all(self._compressed_values[layer_idx], self.value_quantizer)
        return keys, values

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    def to_legacy_cache(self):
        """Convert to legacy tuple format for compatibility."""
        result = []
        for i in range(len(self._compressed_keys)):
            keys, values = self[i]
            result.append((keys, values))
        return tuple(result)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder cache for beam search — not supported with compression."""
        raise NotImplementedError("Beam search not supported with TurboQuant cache")

    # ---------- Compatibility with transformers >= 5.x ----------
    # Transformers expects a `layers` attribute for mask construction.
    # We provide a lightweight shim that exposes get_mask_sizes per layer.

    class _FakeLayer:
        """Minimal shim so cache.layers[i].get_mask_sizes works."""
        def __init__(self, seq_len: int):
            self._seq_len = seq_len
            self.is_sliding = False

        def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
            kv_length = self._seq_len + query_length
            return kv_length, 0

    @property
    def layers(self) -> list:
        return [self._FakeLayer(sl) for sl in self._seq_lengths]

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        """Return (kv_length, kv_offset) for causal mask construction."""
        if layer_idx >= len(self._seq_lengths):
            return query_length, 0
        return self._seq_lengths[layer_idx] + query_length, 0

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    @property
    def is_sliding(self) -> list[bool]:
        return [False] * len(self._seq_lengths)

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def is_initialized(self) -> bool:
        return len(self._compressed_keys) > 0

    @property
    def max_batch_size(self) -> int:
        return 0

    @property
    def max_cache_len(self) -> Optional[int]:
        return None

    def crop(self, max_length: int) -> None:
        pass

    def reset(self) -> None:
        self._compressed_keys.clear()
        self._compressed_values.clear()
        self._seq_lengths.clear()
        self._seen_tokens = 0
