#!/usr/bin/env python3
"""
Benchmark Evaluation

Runs Pass@1 evaluation on GSM8K and MATH datasets.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


def extract_boxed(text: str) -> str:
    """Extract last \\boxed{...} answer from CoT completion."""
    matches = re.findall(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    # Fallback: last number on the last line
    lines = text.strip().splitlines()
    if lines:
        nums = re.findall(r"-?\d+(?:\.\d+)?", lines[-1])
        return nums[-1] if nums else text.strip()
    
    return text.strip()


def extract_final_answer(text: str) -> str:
    """Robustly pull the final answer out of arbitrary model output.

    Mirrors the extractor used by the GRPO reward so eval matches training:
    <answer> tags, \\boxed{}, a **bold** final value, the last $-number, or
    the last number anywhere. Strips $ and commas.
    """
    if not text:
        return ""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    bolds = re.findall(r"\*\*(.+?)\*\*", text)
    if bolds and re.search(r"\d", bolds[-1]):
        return bolds[-1].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        nums = re.findall(r"\$?-?[\d,]+(?:\.\d+)?", lines[-1])
        if nums:
            return nums[-1].replace(",", "")
    all_nums = re.findall(r"\$?-?[\d,]+(?:\.\d+)?", text)
    if all_nums:
        return all_nums[-1].replace(",", "")
    return ""


def extract_answer_from_tags(text: str) -> str:
    """Extract answer from <answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def pass_at_1(generations: List[str], gold: str) -> float:
    """
    Calculate Pass@1 accuracy.
    
    Args:
        generations: List of generated texts
        gold: Gold answer
        
    Returns:
        Fraction correct
    """
    correct = 0
    for gen in generations:
        pred = extract_final_answer(gen)
        if not pred:
            pred = extract_boxed(gen)
        
        try:
            from math_verify import parse, verify
            ok = verify(parse(str(gold)), parse(pred))
        except Exception:
            ok = pred.strip() == str(gold).strip()
        
        if ok:
            correct += 1
    
    return correct / len(generations) if generations else 0.0


def sample_generations(
    model_path: str,
    prompts: List[str],
    n: int = 64,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    backend: str = "transformers",
    gen_chunk: int = 8,
) -> List[List[str]]:
    """
    Sample multiple generations for each prompt.

    backend='transformers' uses HF generate (works with the custom Qwen3.5
    config that vLLM mis-registers as a multimodal model in some envs).
    backend='vllm' uses the faster vLLM engine when your vLLM/transformers
    versions are compatible. Generations are produced in chunks of
    `gen_chunk` to bound VRAM (each chunk loads only `gen_chunk` sequences).
    """
    if backend == "vllm":
        try:
            from vllm import LLM, SamplingParams

            llm = LLM(model=model_path, tensor_parallel_size=1, max_model_len=4096)
            sampling_params = SamplingParams(
                temperature=temperature, top_p=top_p, n=n, max_tokens=max_tokens
            )
            outputs = llm.generate(prompts, sampling_params)
            return [[o.text for o in out.outputs] for out in outputs]
        except Exception as e:
            print(f"vLLM backend failed ({e}); falling back to transformers.")

    # transformers backend (default, env-robust)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    results = []
    for p in prompts:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt"
        ).squeeze(0).to(model.device)
        prompt_len = ids.size(0)
        gens = []
        remaining = n
        while remaining > 0:
            k = min(gen_chunk, remaining)
            with torch.no_grad():
                out = model.generate(
                    ids.unsqueeze(0),
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    num_return_sequences=k,
                    pad_token_id=tok.pad_token_id,
                )
            for seq in out:
                gens.append(tok.decode(seq[prompt_len:], skip_special_tokens=True))
            remaining -= k
        results.append(gens)
    return results


def run_benchmark(
    model_path: str,
    dataset_path: str,
    n_generations: int = 64,
    output_file: str = "eval_results.json",
    backend: str = "transformers",
    gen_chunk: int = 8,
):
    """
    Run benchmark evaluation.
    
    Args:
        model_path: Path to model
        dataset_path: Path to evaluation dataset
        n_generations: Number of generations per prompt
        output_file: Output file for results
    """
    print(f"Loading dataset from {dataset_path}...")
    
    # Load dataset
    prompts = []
    golds = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                # Extract user prompt
                for msg in record["messages"]:
                    if msg["role"] == "user":
                        prompts.append(msg["content"])
                        break
                # Extract gold answer
                gold = record.get("metadata", {}).get("answer", "")
                golds.append(gold)
    
    print(f"Loaded {len(prompts)} problems")
    
    # Sample generations
    print(f"\nGenerating {n_generations} completions per problem...")
    start_time = time.time()
    
    all_generations = sample_generations(
        model_path, prompts, n=n_generations, backend=backend, gen_chunk=gen_chunk
    )
    
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.1f} seconds")
    
    # Calculate Pass@1
    print("\nCalculating Pass@1 accuracy...")
    scores = []
    for gens, gold in zip(all_generations, golds):
        score = pass_at_1(gens, gold)
        scores.append(score)
    
    mean_score = np.mean(scores)
    print(f"Pass@1 accuracy: {mean_score:.4f}")
    
    # Save results
    results = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "n_problems": len(prompts),
        "n_generations": n_generations,
        "mean_pass_at_1": float(mean_score),
        "generation_time": generation_time,
        "individual_scores": scores,
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Evaluation")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--generations", type=int, default=64, help="Number of generations")
    parser.add_argument("--backend", default="transformers", choices=["transformers", "vllm"],
                        help="Generation backend. 'transformers' works with the custom Qwen3.5 "
                             "config; vLLM is faster but crashes if vLLM/transformers mismatch.")
    parser.add_argument("--gen-chunk", type=int, default=8,
                        help="Generations per forward pass (bounds VRAM); only for transformers backend")
    parser.add_argument("--output", default="eval_results.json", help="Output file")
    
    args = parser.parse_args()
    
    run_benchmark(
        model_path=args.model,
        dataset_path=args.dataset,
        n_generations=args.generations,
        output_file=args.output,
        backend=args.backend,
        gen_chunk=args.gen_chunk,
    )


if __name__ == "__main__":
    main()
