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
        pred = extract_answer_from_tags(gen)
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
) -> List[List[str]]:
    """
    Sample multiple generations for each prompt using vLLM.
    
    Args:
        model_path: Path to model
        prompts: List of prompts
        n: Number of generations per prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        
    Returns:
        List of lists of generations
    """
    try:
        from vllm import LLM, SamplingParams
        
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            max_model_len=4096,
        )
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            n=n,
            max_tokens=max_tokens,
        )
        
        outputs = llm.generate(prompts, sampling_params)
        
        # Extract generations
        all_generations = []
        for output in outputs:
            generations = [out.text for out in output.outputs]
            all_generations.append(generations)
        
        return all_generations
        
    except ImportError:
        print("vLLM not available, using placeholder generation")
        return [["Placeholder generation"] * n for _ in prompts]


def run_benchmark(
    model_path: str,
    dataset_path: str,
    n_generations: int = 64,
    output_file: str = "eval_results.json",
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
        model_path, prompts, n=n_generations
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
    parser.add_argument("--output", default="eval_results.json", help="Output file")
    
    args = parser.parse_args()
    
    run_benchmark(
        model_path=args.model,
        dataset_path=args.dataset,
        n_generations=args.generations,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
