# TurboQuant — Paper-to-Code Algorithm Mapping

**Paper:** TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
**arXiv:** 2504.19874

---

## Algorithm 1: MSE-Optimized Quantizer (Stage A)

### Pseudocode (from paper)

```
Input:  x ∈ ℝ^d (input vector), b (bit-width), Π (random rotation matrix)
Output: idx ∈ [2^b]^d (index vector)

1. y ← Π · x                              # rotate
2. For j = 1..d:
     idx_j ← argmin_{k ∈ [2^b]} |y_j - c_k|  # nearest codebook entry
3. Return idx
```

### Dequantization

```
Input:  idx ∈ [2^b]^d, Π
Output: x̃ ∈ ℝ^d

1. For j = 1..d:  ỹ_j ← c_{idx_j}        # lookup codebook
2. x̃ ← Π^T · ỹ                           # inverse rotate
3. Return x̃
```

### Implementation Notes

- **Rotation Π**: Generated via QR decomposition of a d×d matrix with i.i.d. N(0,1) entries. This produces a uniformly random orthogonal matrix. Generated once per head dimension, reused for all tokens.
- **Norm handling**: Paper assumes unit-norm vectors. Real KV vectors are not unit-norm. We normalize before quantization, store the norm, and rescale after dequantization. This adds one float32 per vector.
- **Codebook c_k**: Optimal scalar codebook for Beta(½, (d-1)/2) distribution. This distribution arises because after random rotation, each coordinate of a unit-norm d-dimensional vector follows Beta(½, (d-1)/2) mapped to [-1/√d, +1/√d].

---

## Algorithm 2: Full TurboQuant (Stage A + Stage B)

### Pseudocode

```
Input:  x ∈ ℝ^d, total bit-budget b
Output: (idx, qjl_signs, γ)

Stage A — MSE quantization at (b-1) bits:
1. idx ← Algorithm1(x, b-1, Π)
2. x̃_mse ← Dequantize(idx, Π)

Stage B — QJL residual correction (1 bit):
3. r ← x - x̃_mse                          # residual
4. γ ← ‖r‖₂                               # residual norm
5. qjl_signs ← sign(S · r)                 # sign sketch, S is d×d i.i.d. N(0,1)
6. Return (idx, qjl_signs, γ)
```

### Full Dequantization

```
Input:  (idx, qjl_signs, γ), Π, S
Output: x̃ ∈ ℝ^d

1. x̃_mse ← Dequantize_MSE(idx, Π)
2. qjl_float ← 2 · qjl_signs - 1          # {0,1} → {-1,+1}
3. r̃ ← (√(π/2) / d) · γ · S^T · qjl_float
4. x̃ ← x̃_mse + r̃
5. Return x̃
```

### Inner-Product Estimation (alternative to full dequant)

```
⟨y, x̃⟩ = ⟨y, x̃_mse⟩ + (√(π/2) / d) · γ · ⟨y, S^T · qjl_float⟩
```

This is an unbiased estimator: E[⟨y, x̃⟩] = ⟨y, x⟩

---

## Precomputed Codebook Values

### b=1 (2 centroids)

For dimension d:
```
c = [-√(2/(πd)), +√(2/(πd))]
```

### b=2 (4 centroids)

For dimension d:
```
c = [-1.51/√d, -0.453/√d, +0.453/√d, +1.51/√d]
```

### b=3 (8 centroids), b=4 (16 centroids)

Computed via Lloyd-Max iteration on the Beta(½, (d-1)/2) distribution.
The distribution PDF for a coordinate z of a uniformly random unit vector in ℝ^d is:

```
f(z) ∝ (1 - d·z²)^((d-3)/2)   for z ∈ [-1/√d, +1/√d]
```

This is the Beta(½, (d-1)/2) distribution rescaled from [0,1] to [-1/√d, +1/√d].

---

## Distortion Bounds (from paper)

### MSE (unit-norm vectors)

| Bits | Upper Bound | Typical Measured |
|------|-------------|-----------------|
| 1 | 0.36 | — |
| 2 | 0.117 | — |
| 3 | 0.03 | — |
| 4 | 0.009 | — |

General: D_mse ≤ (√3·π/2) · (1/4^b)

### Inner-Product (unit-norm vectors)

| Bits | Upper Bound |
|------|-------------|
| 1 | 1.57/d |
| 2 | 0.56/d |
| 3 | 0.18/d |
| 4 | 0.047/d |

General: D_prod ≤ (√3·π²·‖y‖²) / (d · 4^b)

---

## Global vs Runtime Objects

| Object | Lifecycle | Storage |
|--------|-----------|---------|
| Rotation matrix Π | Precomputed once per head_dim + seed | In-memory, ~64KB for d=128 |
| Scalar codebook c_k | Precomputed once per (bits, dim) | In-memory or disk cache, tiny |
| QJL projection S | Precomputed once per (dim, seed) | In-memory, ~64KB for d=128 |
| Quantized indices | Per-token, per-head, per-layer | Compressed KV cache |
| Norms | Per-token, per-head, per-layer | Float32, one per vector |
| QJL signs | Per-token, per-head, per-layer | 1 bit per coordinate |
| Residual norm γ | Per-token, per-head, per-layer | Float32, one per vector |

---

## Known Approximations in Our Prototype

1. **Blockwise rotation fallback**: If full d×d rotation matmul is too slow on Apple Silicon, we use block-diagonal rotation with configurable block_size. This weakens the statistical guarantee (coordinates within a block are not fully independent) but may be acceptable in practice.

2. **Dequantize-then-attend**: The paper suggests using the unbiased inner-product estimator directly during attention. Our first prototype uses full dequantization followed by standard attention. This is simpler but may be slower. A direct compressed-attention path is a Phase 2 optimization.

3. **Codebook approximation for b≥3**: We compute Lloyd-Max codebooks numerically rather than using exact analytical solutions (which don't exist in closed form for b≥3). Convergence tolerance is 1e-10.

4. **Shared QJL projection**: We share one S matrix across all layers and heads. The paper says S is data-oblivious so this should be valid, but it means all residuals are projected identically.
