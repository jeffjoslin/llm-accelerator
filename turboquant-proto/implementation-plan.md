# TurboQuant Prototype — Implementation Plan

**Version:** v1
**Owner:** Jeff
**Date:** 2026-03-29

---

## Architecture Overview

The prototype is a Python package (`turboquant-proto/`) with 6 module groups. Each phase builds on the previous one. Everything runs locally on Apple Silicon (24GB Mac Mini Pro) targeting Qwen2.5-1.5B-Instruct as the primary model.

---

## Phase 0: Repo Scaffold + Paper Notes (Foundation)

**What we build:** Project structure, dependencies, configuration system, and a technical notes file mapping paper math to code.

### Repo Layout

```
turboquant-proto/
├── paper_notes/
│   └── algorithm_mapping.md
├── configs/
│   └── default.yaml
├── models/                    # gitignored, for local model caches
├── src/
│   ├── __init__.py
│   ├── bench/
│   │   ├── __init__.py
│   │   ├── run_baseline.py
│   │   └── run_quant.py
│   ├── cache_hooks/
│   │   ├── __init__.py
│   │   └── quantized_cache.py
│   ├── quant/
│   │   ├── __init__.py
│   │   ├── codebook.py
│   │   └── scalar_quantizer.py
│   ├── rotation/
│   │   ├── __init__.py
│   │   └── random_rotation.py
│   ├── qjl/
│   │   ├── __init__.py
│   │   └── qjl_sketch.py
│   └── eval/
│       ├── __init__.py
│       └── compare_outputs.py
├── tests/
│   ├── test_rotation.py
│   ├── test_codebook.py
│   ├── test_quantizer.py
│   ├── test_qjl.py
│   └── test_cache_hooks.py
├── reports/                   # benchmark JSON outputs
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Configuration (`configs/default.yaml`)

```yaml
model: "Qwen/Qwen2.5-1.5B-Instruct"
fallback_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
mode: "baseline"              # baseline | int4_naive | turboquant_mse | turboquant_full
bits: 4
quantize_keys: true
quantize_values: true
rotation_enabled: true
rotation_type: "full"         # full | blockwise
block_size: 64                # for blockwise rotation
qjl_enabled: false            # only true for turboquant_full
seed: 42
max_new_tokens: 256
device: "cpu"                 # or "mps" for Apple Silicon
```

### Algorithm-to-Code Mapping

| Paper Concept | Code Location | Detail |
|---|---|---|
| Random rotation Π | `src/rotation/random_rotation.py` | QR decomposition of random Gaussian matrix. For blockwise: block-diagonal with `block_size` blocks |
| Scalar codebook c_k | `src/quant/codebook.py` | Lloyd-Max optimal for Beta distribution. Precomputed tables for b=1,2,3,4. Values scale as 1/√d |
| Quantize Q_b | `src/quant/scalar_quantizer.py` | argmin_k \|y_j - c_k\| per coordinate |
| Dequantize | `src/quant/scalar_quantizer.py` | x̃ = Π^T · codebook[idx] |
| QJL projection S | `src/qjl/qjl_sketch.py` | i.i.d. N(0,1) entries, d×d matrix |
| Sign sketch | `src/qjl/qjl_sketch.py` | sign(S · r), stored as packed bits |
| Residual norm γ | `src/qjl/qjl_sketch.py` | \|\|r\|\|_2, one scalar per vector |
| QJL reconstruction | `src/qjl/qjl_sketch.py` | x̃_full = x̃_mse + (√(π/2)/d) · γ · S^T · qjl |
| KV cache hook | `src/cache_hooks/quantized_cache.py` | Subclass HF `DynamicCache`, override `update()` and key/value getters |

### Precomputed Codebook Values (from paper, large-d limit)

- **b=1:** `[-√(2/πd), +√(2/πd)]`
- **b=2:** `[-1.51/√d, -0.453/√d, +0.453/√d, +1.51/√d]`
- **b=3, b=4:** Compute via Lloyd-Max on Beta(½, (d-1)/2) distribution at init time, cache to disk

---

## Phase 1: Baseline Inference Harness

**Deliverable:** `src/bench/run_baseline.py` — reproducible baseline benchmark.

### Steps

1. **Load model + tokenizer** from HF with `torch_dtype=torch.float16` (or float32 as fallback on MPS).
2. **Define benchmark prompts:**
   - Short: ~256 token input (a paragraph + question)
   - Medium: ~2048 token input (longer passage)
   - Long: ~4096 token input (if memory allows)
3. **Run generation** with `model.generate()`, `use_cache=True`, capturing:
   - **TTFT:** Time from `generate()` call to first token (hook into `LogitsProcessor` or measure prefill time)
   - **Decode tok/s:** `(total_tokens - 1) / (total_time - ttft)`
   - **Peak memory:** `torch.mps.driver_allocated_memory()` on MPS, or `psutil` RSS on CPU
   - **Output text:** Full decoded output
4. **Repeat 3 times** per prompt, compute mean/stddev.
5. **Save results** to `reports/baseline_{model}_{timestamp}.json`.

### CLI

```bash
python -m src.bench.run_baseline --model qwen2.5-1.5b-instruct --device mps --runs 3
```

### Acceptance Criteria

- Run 3x on the same prompt with same seed produces results within 5% variance for speed metrics.
- JSON file is well-formed and contains all fields.

---

## Phase 2: KV-Cache Interception Layer

**Deliverable:** `src/cache_hooks/quantized_cache.py` — a custom `DynamicCache` subclass.

### Design

```python
class TurboQuantCache(DynamicCache):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.quantizer = build_quantizer(config)
        self._compressed_keys = []
        self._compressed_values = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        # Quantize-on-write
        if self.config.quantize_keys:
            compressed_k = self.quantizer.quantize(key_states)
            self._compressed_keys[layer_idx].append(compressed_k)
        # Same for values
        ...
        # Return dequantized tensors for attention
        return self._get_full_keys(layer_idx), self._get_full_values(layer_idx)
```

### Mode Dispatch

`build_quantizer(config)` returns:
- `mode=baseline` → `NoOpQuantizer` (pass-through, for validation)
- `mode=int4_naive` → `NaiveInt4Quantizer` (simple round-to-nearest 4-bit)
- `mode=turboquant_mse` → `TurboQuantMSEQuantizer`
- `mode=turboquant_full` → `TurboQuantFullQuantizer`

### Acceptance Criteria

- `mode=baseline` wrapper produces byte-identical outputs to vanilla `DynamicCache`.
- `mode=int4_naive` runs without crash.

---

## Phase 3: TurboQuant MSE Stage (Stage A)

**Deliverable:** Working Stage A quantizer in `src/rotation/` and `src/quant/`.

### Rotation Module (`src/rotation/random_rotation.py`)

```python
def generate_rotation_matrix(d, seed=42):
    """Full random orthogonal matrix via QR of Gaussian."""
    rng = np.random.RandomState(seed)
    G = rng.randn(d, d)
    Q, R = np.linalg.qr(G)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return torch.from_numpy(Q).float()

def generate_block_rotation(d, block_size, seed=42):
    """Block-diagonal rotation for efficiency."""
    ...
```

**Key detail:** Rotation matrix is generated **once per head dimension** and reused for all tokens. Precomputed at init. Head dim for Qwen2.5-1.5B = 128, so Π is 128×128.

### Codebook Module (`src/quant/codebook.py`)

Codebooks solve a continuous 1-D k-means problem on Beta(½, (d-1)/2). The Beta distribution arises because after random rotation, each coordinate of a unit-norm vector follows this distribution.

**Norm handling:** Real KV vectors are NOT unit norm. Approach: normalize → quantize → store norm → rescale on dequant.

### Scalar Quantizer (`src/quant/scalar_quantizer.py`)

```python
class TurboQuantMSEQuantizer:
    def quantize(self, x):
        # 1. Store norms
        norms = x.norm(dim=-1, keepdim=True)
        x_unit = x / (norms + 1e-8)
        # 2. Rotate
        y = x_unit @ self.rotation.T
        # 3. Quantize each coordinate to nearest codebook entry
        indices = nearest_codebook_index(y, self.codebook)
        return CompressedTensor(indices=indices, norms=norms)

    def dequantize(self, compressed):
        y_hat = self.codebook[compressed.indices]
        x_hat = y_hat @ self.rotation
        return x_hat * compressed.norms
```

### Index Packing

For b=2,3,4 bits, pack indices into uint8/uint16 tensors. 4-bit = 2 values per byte. 3-bit = 8 values per 3 bytes.

### Acceptance Criteria

- Round-trip MSE on unit vectors ≤ paper bounds (≤0.009 for 4-bit, ≤0.03 for 3-bit).
- Π^T Π ≈ I verified.
- Compressed representation uses measurably less memory than fp16.
- Full generation with `mode=turboquant_mse` at 4-bit produces coherent output.

---

## Phase 4: TurboQuant Full Stage (Stage A + B)

**Deliverable:** `src/qjl/qjl_sketch.py` — QJL residual correction.

### QJL Sketch

```python
class QJLSketch:
    def __init__(self, dim, seed):
        rng = torch.Generator().manual_seed(seed)
        self.S = torch.randn(dim, dim, generator=rng)

    def sketch(self, r):
        gamma = r.norm(dim=-1, keepdim=True)
        projected = r @ self.S.T
        signs = (projected >= 0)
        return signs, gamma

    def reconstruct_residual(self, signs, gamma):
        sign_float = signs.float() * 2 - 1
        r_hat = (math.sqrt(math.pi / 2) / self.S.shape[0]) * gamma * (sign_float @ self.S)
        return r_hat
```

### Combined Quantizer

```python
class TurboQuantFullQuantizer:
    def __init__(self, bits, head_dim, ...):
        self.mse_quantizer = TurboQuantMSEQuantizer(bits - 1, ...)  # (b-1) bits for MSE
        self.qjl = QJLSketch(head_dim, ...)

    def quantize(self, x):
        mse_compressed = self.mse_quantizer.quantize(x)
        x_hat_mse = self.mse_quantizer.dequantize(mse_compressed)
        residual = x - x_hat_mse
        signs, gamma = self.qjl.sketch(residual)
        return FullCompressed(mse=mse_compressed, signs=signs, gamma=gamma)

    def dequantize(self, compressed):
        x_hat_mse = self.mse_quantizer.dequantize(compressed.mse)
        r_hat = self.qjl.reconstruct_residual(compressed.signs, compressed.gamma)
        return x_hat_mse + r_hat
```

**Bit budget:** (b-1) bits for MSE + 1 bit for QJL = b total bits. "4-bit full" = 3-bit MSE + 1-bit QJL.

**Memory concern:** QJL projection matrix S is d×d (128×128 = 64KB). Shared across all heads/layers (data-oblivious per paper).

### Acceptance Criteria

- Unbiasedness: mean inner-product error → 0 over 1000 trials.
- Full mode has lower inner-product error than MSE-only at same total bit budget.
- Generation with `turboquant_full` at 4-bit produces comparable quality to MSE-only at 4-bit.

---

## Phase 5: Benchmark + Compare

**Deliverable:** Benchmark report and recommendation.

### Experiment Matrix

| Run | Mode | Bits | Model |
|-----|------|------|-------|
| 1 | baseline | fp16 | Qwen2.5-1.5B-Instruct |
| 2 | int4_naive | 4 | Qwen2.5-1.5B-Instruct |
| 3 | turboquant_mse | 4 | Qwen2.5-1.5B-Instruct |
| 4 | turboquant_full | 4 | Qwen2.5-1.5B-Instruct |
| 5 | turboquant_mse | 3 | Qwen2.5-1.5B-Instruct |
| 6 | turboquant_full | 3 | Qwen2.5-1.5B-Instruct |

### Metrics

- Peak memory (RSS + GPU/MPS)
- TTFT (ms)
- Decode tokens/sec
- Total generation time
- Output text

### Quality Evaluation (`src/eval/compare_outputs.py`)

- Exact match rate with baseline at greedy decoding
- ROUGE-L between baseline and quantized outputs
- Needle-in-a-haystack retrieval test
- Perplexity proxy (log-likelihood comparison)

### CLI

```bash
python -m src.bench.run_quant --model qwen2.5-1.5b-instruct --mode turboquant_mse --bits 4
python -m src.bench.run_quant --model qwen2.5-1.5b-instruct --mode turboquant_full --bits 3
python -m src.eval.compare_outputs --baseline reports/base.json --candidate reports/tq.json
```

---

## Phase 6: llama.cpp Feasibility Memo

**Deliverable:** Written assessment, no code.

Analyze llama.cpp KV cache types (`ggml_type`), identify insertion points for a new TurboQuant cache type, recommend go/no-go.

---

## Key Design Decisions

1. **Norm handling:** Paper assumes unit vectors. Real KV vectors are not unit norm. We normalize → quantize → store norm → rescale on dequant. One float32 per vector overhead.

2. **Rotation matrix memory:** 128×128 float32 = 64KB. Shared across all tokens in a head. Acceptable.

3. **QJL projection matrix:** Same 64KB per S matrix. Shared across all layers/heads (data-oblivious per paper).

4. **Blockwise rotation fallback:** If full 128×128 matmul is too slow, fall back to block_size=64. Documents divergence from paper.

5. **Bit packing:** Required for actual memory savings. 4-bit = 2 values/byte. 3-bit = 8 values per 3 bytes.

---

## Testing Plan

### Unit Tests

| Test | File | Validates |
|------|------|-----------|
| Rotation orthogonality | `tests/test_rotation.py` | Π^T Π ≈ I, blockwise variant correct |
| Rotation shape | `tests/test_rotation.py` | Output shape matches input shape |
| Codebook values | `tests/test_codebook.py` | Match paper's known entries for b=1, b=2 |
| Codebook optimality | `tests/test_codebook.py` | Lloyd-Max converges, distortion ≤ paper bounds |
| Quantize round-trip MSE | `tests/test_quantizer.py` | MSE on unit vectors ≤ theoretical bound per bit width |
| Index range | `tests/test_quantizer.py` | All indices in [0, 2^b) |
| Index packing | `tests/test_quantizer.py` | Pack → unpack is lossless |
| QJL unbiasedness | `tests/test_qjl.py` | Mean inner-product error → 0 over 1000 trials |
| QJL sketch shape | `tests/test_qjl.py` | Signs are boolean, gamma is scalar per vector |
| Cache no-op equivalence | `tests/test_cache_hooks.py` | Baseline mode = vanilla DynamicCache |
| Cache quantize shape | `tests/test_cache_hooks.py` | Dequantized K/V shapes match expected attention input |

### Integration Tests

| Test | Command | Pass Criteria |
|------|---------|---------------|
| Smoke test (TinyLlama) | `python -m src.bench.run_baseline --model tinyllama` | Completes, JSON saved |
| Smoke test quantized | `python -m src.bench.run_quant --model tinyllama --mode turboquant_mse --bits 4` | Completes, coherent output |
| Baseline reproducibility | Run 3x with same seed | Identical output, metrics within 5% |
| No-op cache equivalence | baseline vs `mode=baseline` cache hook | Token-by-token identical |
| Memory reduction | Compare peak RSS: baseline vs 4-bit TQ | Measurable reduction |

### Running Tests

```bash
# Unit tests
python -m pytest tests/ -v

# Baseline benchmark (quick, TinyLlama)
python -m src.bench.run_baseline --model tinyllama --device cpu --runs 1

# Full benchmark suite
python -m src.bench.run_quant --model qwen2.5-1.5b-instruct --mode turboquant_mse --bits 4 --device mps --runs 3
python -m src.bench.run_quant --model qwen2.5-1.5b-instruct --mode turboquant_full --bits 4 --device mps --runs 3

# Compare outputs
python -m src.eval.compare_outputs --baseline reports/baseline_*.json --candidate reports/tq_*.json
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Full random rotation too expensive | Start with blockwise, document divergence |
| Dequantize-on-read erases speed gains | Phase 1 is correctness-first; optimize later |
| Apple Silicon runtime differences | Log wall-clock + process memory, deterministic prompts, same seed |
| llama.cpp port much harder | Don't begin C/C++ until Python benchmarks justify it |
