"""Quantized inference benchmark — runs generation with TurboQuant KV-cache.

Usage:
    python -m src.bench.run_quant --model qwen2.5-1.5b-instruct --mode turboquant_mse --bits 4
    python -m src.bench.run_quant --model qwen2.5-1.5b-instruct --mode turboquant_full --bits 3
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch

from src.bench.run_baseline import DEFAULT_PROMPTS, MODEL_ALIASES, get_memory_mb, get_peak_memory_mb
from src.cache_hooks.quantized_cache import TurboQuantCache, TurboQuantConfig


def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


def run_single_quant_benchmark(
    model,
    tokenizer,
    prompt: str,
    prompt_label: str,
    max_new_tokens: int,
    device: str,
    seed: int,
    cache_config: TurboQuantConfig,
) -> dict:
    """Run a single generation with quantized KV cache."""
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    mem_before = get_memory_mb()

    # Create quantized cache
    cache = TurboQuantCache(cache_config)

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            past_key_values=cache,
            use_cache=True,
        )

    end = time.perf_counter()
    total_time = end - start

    output_ids = outputs[0][input_len:]
    num_generated = len(output_ids)
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    peak_mem = get_peak_memory_mb(device)
    mem_after = get_memory_mb()

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
        "mode": cache_config.mode,
        "bits": cache_config.bits,
    }


def main():
    parser = argparse.ArgumentParser(description="Run quantized KV-cache inference benchmark")
    parser.add_argument("--model", type=str, default="qwen2.5-1.5b-instruct")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "int4_naive", "turboquant_mse", "turboquant_full"])
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--rotation-type", type=str, default="full", choices=["full", "blockwise"])
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--quantize-keys", action="store_true", default=True)
    parser.add_argument("--quantize-values", action="store_true", default=True)
    parser.add_argument("--no-quantize-keys", dest="quantize_keys", action="store_false")
    parser.add_argument("--no-quantize-values", dest="quantize_values", action="store_false")
    parser.add_argument("--prompts", type=str, nargs="+", default=["short"],
                        choices=["short", "medium"])
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--head-dim", type=int, default=None,
                        help="Override head_dim (auto-detected from model if not set)")
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

    # Auto-detect head_dim from model config
    head_dim = args.head_dim
    if head_dim is None:
        mc = model.config
        if hasattr(mc, "head_dim"):
            head_dim = mc.head_dim
        elif hasattr(mc, "hidden_size") and hasattr(mc, "num_attention_heads"):
            head_dim = mc.hidden_size // mc.num_attention_heads
        else:
            head_dim = 128
            print(f"Warning: could not detect head_dim, using default {head_dim}")
    print(f"Model loaded. head_dim={head_dim}")

    cache_config = TurboQuantConfig(
        mode=args.mode,
        bits=args.bits,
        quantize_keys=args.quantize_keys,
        quantize_values=args.quantize_values,
        rotation_type=args.rotation_type,
        block_size=args.block_size,
        seed=args.seed,
        head_dim=head_dim,
    )

    all_results = []
    for prompt_label in args.prompts:
        prompt = DEFAULT_PROMPTS[prompt_label]
        print(f"\nBenchmark: {prompt_label} prompt, mode={args.mode}, bits={args.bits}, "
              f"{args.runs} runs")

        runs = []
        for i in range(args.runs):
            print(f"  Run {i + 1}/{args.runs}...", end=" ", flush=True)
            result = run_single_quant_benchmark(
                model, tokenizer, prompt, prompt_label,
                args.max_new_tokens, args.device, args.seed, cache_config,
            )
            runs.append(result)
            print(f"{result['decode_tok_per_sec']} tok/s, "
                  f"{result['generated_tokens']} tokens")

        tok_per_sec_vals = [r["decode_tok_per_sec"] for r in runs]
        summary = {
            "prompt_label": prompt_label,
            "num_runs": args.runs,
            "mean_decode_tok_per_sec": round(sum(tok_per_sec_vals) / len(tok_per_sec_vals), 2),
            "runs": runs,
        }
        all_results.append(summary)

    report = {
        "benchmark_type": "quantized",
        "model": model_name,
        "device": args.device,
        "dtype": args.dtype,
        "mode": args.mode,
        "bits": args.bits,
        "quantize_keys": args.quantize_keys,
        "quantize_values": args.quantize_values,
        "rotation_type": args.rotation_type,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace("/", "_").replace(".", "_")
    out_path = Path(args.output_dir) / f"{args.mode}_{args.bits}b_{safe_model}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
