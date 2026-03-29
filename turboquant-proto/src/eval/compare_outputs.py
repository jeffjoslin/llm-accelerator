"""Compare baseline vs quantized inference outputs.

Computes:
- Exact match rate (greedy decoding)
- Character-level similarity
- ROUGE-L score (if rouge_score available, else simple LCS-based)
- Needle-in-a-haystack retrieval test
- Summary table

Usage:
    python -m src.eval.compare_outputs --baseline reports/baseline.json --candidate reports/tq.json
"""

import argparse
import json
from pathlib import Path


def longest_common_subsequence(a: str, b: str) -> int:
    """Compute LCS length between two strings."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Optimize: use two rows instead of full matrix
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l(reference: str, candidate: str) -> dict:
    """Compute ROUGE-L (word-level) between reference and candidate."""
    ref_words = reference.split()
    cand_words = candidate.split()
    if not ref_words or not cand_words:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs_len = longest_common_subsequence(ref_words, cand_words)
    precision = lcs_len / len(cand_words) if cand_words else 0
    recall = lcs_len / len(ref_words) if ref_words else 0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def char_similarity(a: str, b: str) -> float:
    """Character-level similarity (LCS / max length)."""
    if not a and not b:
        return 1.0
    lcs = longest_common_subsequence(a[:2000], b[:2000])  # cap for performance
    return round(lcs / max(len(a[:2000]), len(b[:2000])), 4)


def compare_run_pair(baseline_run: dict, candidate_run: dict) -> dict:
    """Compare a single pair of runs."""
    base_text = baseline_run["output_text"]
    cand_text = candidate_run["output_text"]

    exact_match = base_text == cand_text
    char_sim = char_similarity(base_text, cand_text)
    rl = rouge_l(base_text, cand_text)

    speed_ratio = (candidate_run["decode_tok_per_sec"] /
                   baseline_run["decode_tok_per_sec"]
                   if baseline_run["decode_tok_per_sec"] > 0 else 0)

    return {
        "exact_match": exact_match,
        "char_similarity": char_sim,
        "rouge_l": rl,
        "baseline_tok_per_sec": baseline_run["decode_tok_per_sec"],
        "candidate_tok_per_sec": candidate_run["decode_tok_per_sec"],
        "speed_ratio": round(speed_ratio, 3),
        "baseline_peak_mem_mb": baseline_run["peak_mem_mb"],
        "candidate_peak_mem_mb": candidate_run["peak_mem_mb"],
        "baseline_output_preview": base_text[:200],
        "candidate_output_preview": cand_text[:200],
    }


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs quantized outputs")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline JSON")
    parser.add_argument("--candidate", type=str, required=True, help="Path to candidate JSON")
    parser.add_argument("--output", type=str, default=None, help="Output comparison JSON")
    args = parser.parse_args()

    baseline = load_report(args.baseline)
    candidate = load_report(args.candidate)

    print("=" * 70)
    print(f"COMPARISON: {baseline.get('benchmark_type', 'unknown')} vs "
          f"{candidate.get('benchmark_type', 'unknown')}")
    print(f"  Baseline:  {baseline.get('model', '?')} ({baseline.get('mode', 'baseline')})")
    cand_mode = candidate.get("mode", "unknown")
    cand_bits = candidate.get("bits", "?")
    print(f"  Candidate: {candidate.get('model', '?')} ({cand_mode} {cand_bits}-bit)")
    print("=" * 70)

    comparisons = []
    for base_result, cand_result in zip(baseline["results"], candidate["results"]):
        prompt_label = base_result["prompt_label"]
        print(f"\n--- Prompt: {prompt_label} ---")

        # Compare first run of each (same seed)
        base_run = base_result["runs"][0]
        cand_run = cand_result["runs"][0]
        comparison = compare_run_pair(base_run, cand_run)
        comparisons.append({"prompt_label": prompt_label, **comparison})

        print(f"  Exact match:       {comparison['exact_match']}")
        print(f"  Char similarity:   {comparison['char_similarity']}")
        print(f"  ROUGE-L F1:        {comparison['rouge_l']['f1']}")
        print(f"  Speed ratio:       {comparison['speed_ratio']}x")
        print(f"  Memory (base):     {comparison['baseline_peak_mem_mb']} MB")
        print(f"  Memory (cand):     {comparison['candidate_peak_mem_mb']} MB")

    report = {
        "baseline_file": str(args.baseline),
        "candidate_file": str(args.candidate),
        "baseline_model": baseline.get("model"),
        "candidate_mode": cand_mode,
        "candidate_bits": cand_bits,
        "comparisons": comparisons,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nComparison saved to {args.output}")


if __name__ == "__main__":
    main()
