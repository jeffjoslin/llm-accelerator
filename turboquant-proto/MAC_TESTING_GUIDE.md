# TurboQuant Prototype — Mac Testing Guide

**Target Machine:** Mac Mini Pro, 24 GB unified memory, Apple Silicon
**Agent Instructions:** Follow each section in order. Report results for each step before proceeding to the next.

---

## 0. Environment Setup

### 0.1 Clone and enter the repo

```bash
cd llm-accelerator
git fetch origin claude/plan-turboquant-prototype-RBW89
git checkout claude/plan-turboquant-prototype-RBW89
cd turboquant-proto
```

### 0.2 Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 0.3 Install dependencies

```bash
pip install -r requirements.txt
```

**Verify PyTorch MPS support:**
```bash
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

Expected output: `MPS available: True`

### 0.4 Run unit tests

```bash
python -m pytest tests/ -v
```

**Expected result:** All 69 tests pass. If any fail, stop and report which tests failed with full output before proceeding.

---

## 1. Experiment A — Smoke Test (TinyLlama)

**Goal:** Prove cache interception works end-to-end with a real model.

### 1.1 Baseline smoke test

```bash
python -m src.bench.run_baseline \
    --model tinyllama \
    --device mps \
    --runs 1 \
    --max-new-tokens 64 \
    --dtype float32 \
    --prompts short
```

**Report:**
- Did it complete? (yes/no)
- Path to output JSON
- Generated tokens count
- Decode tok/s
- Peak memory MB
- First 100 chars of output text

**Note:** Use `--dtype float32` for TinyLlama on MPS. If MPS fails, fall back to `--device cpu`.

### 1.2 Quantized smoke test — TurboQuant MSE 4-bit

```bash
python -m src.bench.run_quant \
    --model tinyllama \
    --device mps \
    --runs 1 \
    --max-new-tokens 64 \
    --dtype float32 \
    --mode turboquant_mse \
    --bits 4 \
    --prompts short
```

**Report:**
- Did it complete? (yes/no)
- Path to output JSON
- Is the output coherent English? (yes/no/partial)
- Decode tok/s
- Peak memory MB

### 1.3 Quantized smoke test — TurboQuant Full 4-bit

```bash
python -m src.bench.run_quant \
    --model tinyllama \
    --device mps \
    --runs 1 \
    --max-new-tokens 64 \
    --dtype float32 \
    --mode turboquant_full \
    --bits 4 \
    --prompts short
```

**Report:** Same as 1.2.

### 1.4 Quantized smoke test — Naive Int4

```bash
python -m src.bench.run_quant \
    --model tinyllama \
    --device mps \
    --runs 1 \
    --max-new-tokens 64 \
    --dtype float32 \
    --mode int4_naive \
    --bits 4 \
    --prompts short
```

**Report:** Same as 1.2.

### 1.5 No-op cache equivalence test

Run baseline mode through the cache wrapper and compare with vanilla baseline:

```bash
python -m src.bench.run_quant \
    --model tinyllama \
    --device mps \
    --runs 1 \
    --max-new-tokens 64 \
    --dtype float32 \
    --mode baseline \
    --prompts short
```

**Verification:** Compare the `output_text` field in both JSONs (step 1.1 and this step). They must be **identical**. Report:
- Exact match? (yes/no)
- If no, paste first diff

---

## 2. Experiment B — Real Prototype (Qwen2.5-1.5B-Instruct)

**Goal:** Produce the actual before/after benchmark numbers.

### 2.1 Baseline

```bash
python -m src.bench.run_baseline \
    --model qwen2.5-1.5b-instruct \
    --device mps \
    --runs 3 \
    --max-new-tokens 256 \
    --dtype float16 \
    --prompts short medium
```

**Report:**
- Mean decode tok/s (short prompt)
- Mean decode tok/s (medium prompt)
- Peak memory MB
- First 200 chars of output text (short prompt)

**Troubleshooting:** If you get OOM with float16, try `--dtype float32` (uses more memory but may be more stable on MPS) or fall back to `--device cpu`.

### 2.2 Naive 4-bit KV

```bash
python -m src.bench.run_quant \
    --model qwen2.5-1.5b-instruct \
    --device mps \
    --runs 3 \
    --max-new-tokens 256 \
    --dtype float16 \
    --mode int4_naive \
    --bits 4 \
    --prompts short medium
```

### 2.3 TurboQuant MSE 4-bit

```bash
python -m src.bench.run_quant \
    --model qwen2.5-1.5b-instruct \
    --device mps \
    --runs 3 \
    --max-new-tokens 256 \
    --dtype float16 \
    --mode turboquant_mse \
    --bits 4 \
    --prompts short medium
```

### 2.4 TurboQuant Full 4-bit

```bash
python -m src.bench.run_quant \
    --model qwen2.5-1.5b-instruct \
    --device mps \
    --runs 3 \
    --max-new-tokens 256 \
    --dtype float16 \
    --mode turboquant_full \
    --bits 4 \
    --prompts short medium
```

### 2.5 TurboQuant MSE 3-bit

```bash
python -m src.bench.run_quant \
    --model qwen2.5-1.5b-instruct \
    --device mps \
    --runs 3 \
    --max-new-tokens 256 \
    --dtype float16 \
    --mode turboquant_mse \
    --bits 3 \
    --prompts short medium
```

### 2.6 TurboQuant Full 3-bit

```bash
python -m src.bench.run_quant \
    --model qwen2.5-1.5b-instruct \
    --device mps \
    --runs 3 \
    --max-new-tokens 256 \
    --dtype float16 \
    --mode turboquant_full \
    --bits 3 \
    --prompts short medium
```

### 2.7 Report for each run (2.1–2.6)

For each of the 6 runs above, report:
- Mode and bits
- Mean decode tok/s (short)
- Mean decode tok/s (medium)
- Peak memory MB
- Output text quality (coherent/degraded/garbage)
- Path to output JSON

---

## 3. Quality Comparison

After all experiment B runs are complete, compare outputs.

### 3.1 Compare each mode against baseline

Run for each candidate JSON against the baseline JSON:

```bash
# Example for TurboQuant MSE 4-bit
python -m src.eval.compare_outputs \
    --baseline reports/baseline_qwen2_5-1_5b-instruct_*.json \
    --candidate reports/turboquant_mse_4b_qwen2_5-1_5b-instruct_*.json \
    --output reports/comparison_mse_4b.json
```

Repeat for all 5 candidate modes (int4_naive, turboquant_mse 4b, turboquant_full 4b, turboquant_mse 3b, turboquant_full 3b).

**Report for each comparison:**
- Exact match? (yes/no)
- Character similarity
- ROUGE-L F1
- Speed ratio (candidate / baseline)
- Memory comparison

---

## 4. Experiment C — Long-Context Stress Test

**Goal:** Test whether compressed cache extends feasible context length.

### 4.1 Long prompt baseline

```bash
python -m src.bench.run_baseline \
    --model qwen2.5-1.5b-instruct \
    --device mps \
    --runs 1 \
    --max-new-tokens 128 \
    --dtype float16 \
    --prompts medium
```

### 4.2 Long prompt with TurboQuant MSE 3-bit

```bash
python -m src.bench.run_quant \
    --model qwen2.5-1.5b-instruct \
    --device mps \
    --runs 1 \
    --max-new-tokens 128 \
    --dtype float16 \
    --mode turboquant_mse \
    --bits 3 \
    --prompts medium
```

**Report:**
- Did both complete? (yes/no)
- Memory comparison
- Any OOM errors?
- If baseline OOMs but quantized doesn't, note this as a key finding

---

## 5. Summary Table

After all experiments, fill in this table:

| Mode | Bits | Decode tok/s (short) | Decode tok/s (medium) | Peak Mem MB | ROUGE-L F1 vs baseline | Output Quality |
|------|------|---------------------|-----------------------|-------------|----------------------|----------------|
| baseline | fp16 | | | | 1.000 | reference |
| int4_naive | 4 | | | | | |
| turboquant_mse | 4 | | | | | |
| turboquant_full | 4 | | | | | |
| turboquant_mse | 3 | | | | | |
| turboquant_full | 3 | | | | | |

---

## 6. Expected Issues and Troubleshooting

### MPS Compatibility

If MPS operations fail with errors like "not implemented for MPS":
1. First try `--dtype float32` (some MPS ops don't support float16)
2. If still failing, fall back to `--device cpu`
3. Report which operations failed

### OOM on MPS

If you get out-of-memory errors:
1. Close other applications
2. Try `--dtype float32` first (counterintuitively, this sometimes helps on MPS)
3. Reduce `--max-new-tokens` to 128
4. Fall back to `--device cpu`
5. Try the TinyLlama model instead

### Model Download Issues

Models are downloaded from HuggingFace on first run. If download fails:
```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### Slow Generation

The quantized modes will be slower than baseline because:
- Dequantize-on-read adds overhead per attention computation
- This is expected — Phase 1 is about correctness, not speed
- If quantized mode is >10x slower, report this as a concern

### Cache Hook Compatibility

If `model.generate()` fails with the custom cache, the issue is likely the cache interface. Report the full traceback. Common fixes:
- Check if the model uses grouped-query attention (GQA) — head counts may differ for K vs V
- Check if `cache_kwargs` contains expected keys

---

## 7. Final Deliverable

After completing all sections, produce a summary with:

1. **All JSON report files** in `reports/`
2. **Filled summary table** from Section 5
3. **Recommendation** — one of:
   - **A:** Continue with Python optimization only
   - **B:** Port to llama.cpp (if memory savings are significant and quality is good)
   - **C:** Re-scope to MSE-only quantization (if QJL adds overhead without quality benefit)
   - **D:** Stop (if performance/quality tradeoff is not good enough on local hardware)
4. **List of any issues encountered** and how they were resolved
5. **List of any code changes** made during testing (if any)

Commit all report JSONs and the filled summary table to the repo.
