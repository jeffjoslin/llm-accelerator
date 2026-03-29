"""Baseline inference benchmark — no KV-cache quantization.

Loads a model, runs generation with standard caching, and records
TTFT, decode speed, peak memory, and output text.

Usage:
    python -m src.bench.run_baseline --model qwen2.5-1.5b-instruct --device mps --runs 3
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch
import yaml

# Model name aliases for convenience
MODEL_ALIASES = {
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# Default benchmark prompts
DEFAULT_PROMPTS = {
    "short": (
        "Explain the theory of general relativity in simple terms, covering the key "
        "concepts of spacetime curvature, gravitational time dilation, and the "
        "equivalence principle. Provide at least one real-world example for each concept."
    ),
    "medium": (
        "Write a comprehensive technical overview of how modern transformer-based "
        "language models work. Cover the following topics in detail:\n\n"
        "1. The attention mechanism and how it differs from RNNs\n"
        "2. Multi-head attention and why it helps\n"
        "3. Positional encoding approaches (sinusoidal vs rotary)\n"
        "4. The role of layer normalization\n"
        "5. The feed-forward network layers\n"
        "6. How the KV-cache works during autoregressive generation\n"
        "7. Common optimization techniques (FlashAttention, quantization, pruning)\n"
        "8. Scaling laws and what they tell us about model size vs performance\n\n"
        "For each topic, explain the mathematical intuition and provide practical "
        "examples. Discuss trade-offs and current open problems in the field. "
        "This should be written at the level of a graduate student in machine learning "
        "who has basic familiarity with neural networks but wants to deeply understand "
        "transformers. Include relevant equations where helpful.\n\n"
        "Begin with a high-level overview of the transformer architecture, then dive "
        "into each component systematically."
    ),
}


class TTFTCallback:
    """Callback to measure time to first token."""

    def __init__(self):
        self.first_token_time = None
        self.start_time = None

    def __call__(self, *args, **kwargs):
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()


def get_memory_mb() -> float:
    """Get current process RSS in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def get_peak_memory_mb(device: str) -> float:
    """Get peak memory depending on device."""
    if device == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
        return torch.mps.driver_allocated_memory() / (1024 * 1024)
    elif device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return get_memory_mb()


def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


def run_single_benchmark(
    model,
    tokenizer,
    prompt: str,
    prompt_label: str,
    max_new_tokens: int,
    device: str,
    seed: int,
) -> dict:
    """Run a single generation and measure metrics."""
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    mem_before = get_memory_mb()

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    end = time.perf_counter()
    total_time = end - start

    output_ids = outputs[0][input_len:]
    num_generated = len(output_ids)
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    peak_mem = get_peak_memory_mb(device)
    mem_after = get_memory_mb()

    # Approximate TTFT as fraction of total time proportional to prefill
    # (Proper TTFT measurement requires hooks; this is a reasonable proxy)
    ttft_approx = total_time * (input_len / (input_len + num_generated))

    decode_time = total_time - ttft_approx
    decode_tok_per_sec = num_generated / decode_time if decode_time > 0 else 0

    return {
        "prompt_label": prompt_label,
        "prompt_tokens": input_len,
        "max_new_tokens": max_new_tokens,
        "generated_tokens": num_generated,
        "total_time_s": round(total_time, 4),
        "ttft_approx_s": round(ttft_approx, 4),
        "decode_tok_per_sec": round(decode_tok_per_sec, 2),
        "mem_before_mb": round(mem_before, 1),
        "mem_after_mb": round(mem_after, 1),
        "peak_mem_mb": round(peak_mem, 1),
        "output_text": output_text,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Run baseline inference benchmark")
    parser.add_argument("--model", type=str, default="qwen2.5-1.5b-instruct")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompts", type=str, nargs="+", default=["short"],
                        choices=["short", "medium"])
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"])
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = resolve_model_name(args.model)
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model: {model_name} (dtype={args.dtype}, device={args.device})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if args.device != "cpu":
        model = model.to(args.device)
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    all_results = []
    for prompt_label in args.prompts:
        prompt = DEFAULT_PROMPTS[prompt_label]
        print(f"\nBenchmark: {prompt_label} prompt, {args.runs} runs")

        runs = []
        for i in range(args.runs):
            print(f"  Run {i + 1}/{args.runs}...", end=" ", flush=True)
            result = run_single_benchmark(
                model, tokenizer, prompt, prompt_label,
                args.max_new_tokens, args.device, args.seed,
            )
            runs.append(result)
            print(f"{result['decode_tok_per_sec']} tok/s, "
                  f"{result['generated_tokens']} tokens")

        # Compute stats across runs
        tok_per_sec_vals = [r["decode_tok_per_sec"] for r in runs]
        summary = {
            "prompt_label": prompt_label,
            "num_runs": args.runs,
            "mean_decode_tok_per_sec": round(sum(tok_per_sec_vals) / len(tok_per_sec_vals), 2),
            "runs": runs,
        }
        all_results.append(summary)

    report = {
        "benchmark_type": "baseline",
        "model": model_name,
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace("/", "_").replace(".", "_")
    out_path = Path(args.output_dir) / f"baseline_{safe_model}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
