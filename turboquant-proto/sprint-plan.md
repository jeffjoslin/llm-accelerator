# TurboQuant Prototype — Sprint Plan

**Start Date:** 2026-03-29
**Owner:** Jeff

---

## Sprint Overview

This sprint executes the TurboQuant prototype from repo scaffold through benchmark report. Work is organized into 18 tasks across 6 phases. Each task has clear inputs, outputs, and acceptance criteria.

Dependencies flow top-to-bottom — each task lists what must be done before it can start.

---

## Task Breakdown

### Phase 0: Foundation

#### Task 0.1 — Repo Scaffold
- **Depends on:** Nothing
- **Work:**
  - Create full directory structure (`src/`, `tests/`, `configs/`, `paper_notes/`, `reports/`)
  - Create all `__init__.py` files
  - Write `requirements.txt` and `pyproject.toml`
  - Write `configs/default.yaml`
  - Add `.gitignore` (models/, reports/*.json, __pycache__, etc.)
- **Output:** Importable Python package structure
- **Acceptance:** `python -c "import src"` succeeds

#### Task 0.2 — Paper Notes / Algorithm Mapping
- **Depends on:** Nothing (can run parallel with 0.1)
- **Work:**
  - Write `paper_notes/algorithm_mapping.md`
  - Document Algorithm 1 (MSE quantizer) pseudocode
  - Document Algorithm 2 (QJL correction) pseudocode
  - Document precomputed codebook values for b=1,2
  - List known approximations we're making vs. the paper
- **Output:** `paper_notes/algorithm_mapping.md`
- **Acceptance:** Document covers all 7 rows of the algorithm-to-code mapping table

---

### Phase 1: Baseline Harness

#### Task 1.1 — Baseline Benchmark Runner
- **Depends on:** Task 0.1
- **Work:**
  - Implement `src/bench/run_baseline.py` with CLI via argparse
  - Load model/tokenizer from HF
  - Define 3 benchmark prompts (short/medium/long)
  - Measure TTFT, decode tok/s, peak memory, output text
  - Support `--runs N` for repeated measurement with mean/stddev
  - Save results as JSON to `reports/`
- **Output:** Working CLI: `python -m src.bench.run_baseline --model <name> --device <dev> --runs 3`
- **Acceptance:**
  - Runs to completion on CPU with TinyLlama
  - JSON output is well-formed with all metric fields
  - 3 runs produce speed metrics within 5% variance

---

### Phase 2: KV-Cache Interception

#### Task 2.1 — NoOp Cache Wrapper
- **Depends on:** Task 0.1
- **Work:**
  - Implement `TurboQuantCache(DynamicCache)` in `src/cache_hooks/quantized_cache.py`
  - Implement `NoOpQuantizer` (pass-through)
  - Implement `build_quantizer(config)` dispatch function
  - Wire cache into `model.generate()` via `past_key_values=` argument
- **Output:** Cache wrapper that passes through K/V unchanged
- **Acceptance:** Baseline mode produces token-identical output vs. vanilla DynamicCache

#### Task 2.2 — Naive Int4 Quantizer
- **Depends on:** Task 2.1
- **Work:**
  - Implement `NaiveInt4Quantizer` (simple per-tensor min/max → 4-bit rounding)
  - Wire into `build_quantizer` for `mode=int4_naive`
- **Output:** Simple quantizer baseline for comparison
- **Acceptance:** Generation completes without crash; output is recognizable English

#### Task 2.3 — Cache Hook Tests
- **Depends on:** Task 2.1, 2.2
- **Work:**
  - Write `tests/test_cache_hooks.py`
  - Test no-op equivalence
  - Test quantized K/V shape preservation
  - Test that update() returns correct tensor shapes
- **Output:** Passing test suite
- **Acceptance:** `pytest tests/test_cache_hooks.py` all green

---

### Phase 3: TurboQuant MSE Stage

#### Task 3.1 — Rotation Module
- **Depends on:** Task 0.1
- **Work:**
  - Implement `generate_rotation_matrix(d, seed)` via QR decomposition
  - Implement `generate_block_rotation(d, block_size, seed)` for blockwise variant
  - Implement inverse rotation (just transpose)
- **Output:** `src/rotation/random_rotation.py`
- **Acceptance:** Π^T Π ≈ I (within float32 tolerance)

#### Task 3.2 — Rotation Tests
- **Depends on:** Task 3.1
- **Work:**
  - Write `tests/test_rotation.py`
  - Test orthogonality for full and blockwise
  - Test shape preservation
  - Test determinism (same seed → same matrix)
- **Output:** Passing tests
- **Acceptance:** `pytest tests/test_rotation.py` all green

#### Task 3.3 — Codebook Module
- **Depends on:** Task 0.1
- **Work:**
  - Implement closed-form codebooks for b=1, b=2
  - Implement Lloyd-Max iteration for b=3, b=4 on Beta(½, (d-1)/2)
  - Implement codebook caching to disk
  - Implement `nearest_codebook_index()` function
- **Output:** `src/quant/codebook.py`
- **Acceptance:** b=1 codebook matches `±√(2/πd)`; b=2 matches `±0.453/√d, ±1.51/√d`

#### Task 3.4 — Codebook Tests
- **Depends on:** Task 3.3
- **Work:**
  - Write `tests/test_codebook.py`
  - Test known values for b=1, b=2
  - Test Lloyd-Max convergence for b=3, b=4
  - Test distortion bounds match paper
- **Output:** Passing tests
- **Acceptance:** `pytest tests/test_codebook.py` all green

#### Task 3.5 — MSE Scalar Quantizer
- **Depends on:** Task 3.1, Task 3.3
- **Work:**
  - Implement `TurboQuantMSEQuantizer` class
  - Quantize path: normalize → rotate → nearest codebook → store indices + norms
  - Dequantize path: lookup → inverse rotate → rescale
  - Implement index packing for b=2,3,4 (bit-level packing into uint8)
  - Implement index unpacking
- **Output:** `src/quant/scalar_quantizer.py`
- **Acceptance:** Round-trip MSE on unit vectors ≤ paper bounds

#### Task 3.6 — Quantizer Tests
- **Depends on:** Task 3.5
- **Work:**
  - Write `tests/test_quantizer.py`
  - Test round-trip MSE bounds for each bit width
  - Test index range [0, 2^b)
  - Test pack → unpack lossless
  - Test shape preservation through quantize/dequantize
- **Output:** Passing tests
- **Acceptance:** `pytest tests/test_quantizer.py` all green

#### Task 3.7 — Wire MSE Quantizer into Cache
- **Depends on:** Task 2.1, Task 3.5
- **Work:**
  - Wire `TurboQuantMSEQuantizer` into `build_quantizer` for `mode=turboquant_mse`
  - Ensure cache update/read cycle works end-to-end
  - Handle per-layer rotation matrix init
- **Output:** Working `mode=turboquant_mse` in cache hook
- **Acceptance:** Generation with `turboquant_mse` 4-bit produces coherent output

---

### Phase 4: QJL Residual Correction

#### Task 4.1 — QJL Sketch Module
- **Depends on:** Task 0.1
- **Work:**
  - Implement `QJLSketch` class with random Gaussian projection S
  - Implement `sketch(r)` → (signs, gamma)
  - Implement `reconstruct_residual(signs, gamma)` → r̂
  - Use formula: r̂ = (√(π/2)/d) · γ · S^T · qjl
  - Handle memory: share S matrix across layers/heads
- **Output:** `src/qjl/qjl_sketch.py`
- **Acceptance:** Reconstruction is unbiased (mean IP error → 0 over many trials)

#### Task 4.2 — QJL Tests
- **Depends on:** Task 4.1
- **Work:**
  - Write `tests/test_qjl.py`
  - Test unbiasedness over 1000 random vector pairs
  - Test sign shape (boolean tensor)
  - Test gamma is scalar per vector
  - Test reconstruction shape matches input
- **Output:** Passing tests
- **Acceptance:** `pytest tests/test_qjl.py` all green

#### Task 4.3 — Full TurboQuant Quantizer
- **Depends on:** Task 3.5, Task 4.1
- **Work:**
  - Implement `TurboQuantFullQuantizer` combining MSE (b-1 bits) + QJL (1 bit)
  - Quantize: MSE quantize → compute residual → QJL sketch
  - Dequantize: MSE dequant + QJL residual reconstruction
  - Wire into `build_quantizer` for `mode=turboquant_full`
- **Output:** Working full quantizer
- **Acceptance:** Inner-product error lower than MSE-only at same total bit budget

#### Task 4.4 — Wire Full Quantizer into Cache
- **Depends on:** Task 4.3, Task 2.1
- **Work:**
  - Integrate `TurboQuantFullQuantizer` into cache hook
  - Verify end-to-end generation
- **Output:** Working `mode=turboquant_full` in cache hook
- **Acceptance:** Generation completes; output quality comparable to MSE-only

---

### Phase 5: Benchmark + Eval

#### Task 5.1 — Quantized Benchmark Runner
- **Depends on:** Task 1.1, Task 3.7, Task 4.4
- **Work:**
  - Implement `src/bench/run_quant.py` with CLI
  - Accept `--mode` and `--bits` flags
  - Reuse same prompt set and metrics as baseline runner
  - Save results to `reports/`
- **Output:** Working CLI: `python -m src.bench.run_quant --model <name> --mode <mode> --bits <b>`
- **Acceptance:** All 6 experiment runs complete and produce JSON

#### Task 5.2 — Output Comparison Tool
- **Depends on:** Task 5.1
- **Work:**
  - Implement `src/eval/compare_outputs.py`
  - Compute exact match rate, ROUGE-L, character-level diff
  - Implement needle-in-a-haystack test (embed fact, query retrieval)
  - Generate summary table (markdown or terminal)
- **Output:** Working CLI: `python -m src.eval.compare_outputs --baseline <json> --candidate <json>`
- **Acceptance:** Produces readable comparison report

#### Task 5.3 — Run Full Experiment Suite + Report
- **Depends on:** Task 5.1, Task 5.2
- **Work:**
  - Run all 6 experiments on Qwen2.5-1.5B-Instruct
  - Run smoke test on TinyLlama first
  - Run long-context stress test (4096+ tokens if possible)
  - Compile results into final report
  - Produce recommendation: A (Python only), B (port to llama.cpp), C (MSE-only), or D (stop)
- **Output:** `reports/final_report.md`
- **Acceptance:** Report contains all metrics, comparison table, and clear recommendation

---

### Phase 6: llama.cpp Assessment

#### Task 6.1 — Feasibility Memo
- **Depends on:** Task 5.3
- **Work:**
  - Review llama.cpp KV cache types
  - Identify insertion points (cache storage, write/read path, attention kernels, CLI flags)
  - Estimate effort for native integration vs. bridge approach
  - Write recommendation
- **Output:** `paper_notes/llamacpp_feasibility.md`
- **Acceptance:** Document addresses all insertion points and provides clear go/no-go

---

## Dependency Graph

```
Phase 0:  [0.1 Scaffold] ──┬──→ [0.2 Paper Notes]
                            │
Phase 1:  ├──→ [1.1 Baseline Runner]
          │
Phase 2:  ├──→ [2.1 NoOp Cache] → [2.2 Naive Int4] → [2.3 Cache Tests]
          │
Phase 3:  ├──→ [3.1 Rotation] → [3.2 Rotation Tests]
          ├──→ [3.3 Codebook] → [3.4 Codebook Tests]
          │         │                    │
          │         └──→ [3.5 MSE Quantizer] → [3.6 Quantizer Tests]
          │                      │
          │                      └──→ [3.7 Wire MSE into Cache]
          │
Phase 4:  ├──→ [4.1 QJL Sketch] → [4.2 QJL Tests]
          │         │
          │         └──→ [4.3 Full Quantizer] → [4.4 Wire Full into Cache]
          │
Phase 5:  └──→ [5.1 Quant Runner] → [5.2 Compare Tool] → [5.3 Full Experiment]
                                                                   │
Phase 6:                                                    [6.1 Feasibility Memo]
```

## Parallelism Opportunities

These task groups can be developed in parallel:

- **Group A (independent core math):** 3.1 Rotation, 3.3 Codebook, 4.1 QJL Sketch
- **Group B (cache infrastructure):** 2.1 NoOp Cache, 2.2 Naive Int4
- **Group C (benchmark infra):** 1.1 Baseline Runner

All three groups depend only on Task 0.1 (scaffold).

---

## Execution Order (Sequential Path)

For single-developer execution, the recommended order is:

| Step | Task | Estimated Effort |
|------|------|-----------------|
| 1 | 0.1 Scaffold | Small |
| 2 | 0.2 Paper Notes | Small |
| 3 | 3.1 Rotation Module | Medium |
| 4 | 3.2 Rotation Tests | Small |
| 5 | 3.3 Codebook Module | Medium |
| 6 | 3.4 Codebook Tests | Small |
| 7 | 3.5 MSE Scalar Quantizer | Large |
| 8 | 3.6 Quantizer Tests | Medium |
| 9 | 4.1 QJL Sketch Module | Medium |
| 10 | 4.2 QJL Tests | Small |
| 11 | 2.1 NoOp Cache Wrapper | Medium |
| 12 | 2.2 Naive Int4 Quantizer | Small |
| 13 | 2.3 Cache Hook Tests | Small |
| 14 | 3.7 Wire MSE into Cache | Medium |
| 15 | 4.3 Full TurboQuant Quantizer | Medium |
| 16 | 4.4 Wire Full into Cache | Small |
| 17 | 1.1 Baseline Benchmark Runner | Medium |
| 18 | 5.1 Quantized Benchmark Runner | Medium |
| 19 | 5.2 Output Comparison Tool | Medium |
| 20 | 5.3 Run Experiments + Report | Large |
| 21 | 6.1 llama.cpp Feasibility Memo | Medium |

---

## Validation Checkpoints

After each phase, run the full test suite to confirm no regressions:

```bash
# After Phase 0
python -c "import src"

# After Phase 1
python -m src.bench.run_baseline --model tinyllama --device cpu --runs 1

# After Phase 2
python -m pytest tests/test_cache_hooks.py -v

# After Phase 3
python -m pytest tests/test_rotation.py tests/test_codebook.py tests/test_quantizer.py -v
python -m src.bench.run_quant --model tinyllama --mode turboquant_mse --bits 4 --device cpu

# After Phase 4
python -m pytest tests/ -v
python -m src.bench.run_quant --model tinyllama --mode turboquant_full --bits 4 --device cpu

# After Phase 5
python -m src.eval.compare_outputs --baseline reports/baseline_*.json --candidate reports/tq_*.json
```

---

## Definition of Done

The sprint is complete when:

1. All 21 tasks are done
2. `pytest tests/ -v` is all green
3. All 6 benchmark experiments have run on Qwen2.5-1.5B-Instruct
4. `reports/final_report.md` exists with metrics table and recommendation
5. Recommendation is one of: A (Python only), B (port to llama.cpp), C (MSE-only), D (stop)
