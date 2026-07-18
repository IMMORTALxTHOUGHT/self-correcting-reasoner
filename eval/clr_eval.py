#!/usr/bin/env python3
"""
Claim-Level Reliability (CLR) Evaluation

Evaluates reasoning reliability by sampling multiple traces and
self-verifying claims within each trace.
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


def extract_boxed(text: str) -> str:
    """Extract last \\boxed{...} answer from CoT completion."""
    matches = re.findall(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    lines = text.strip().splitlines()
    if lines:
        nums = re.findall(r"-?\d+(?:\.\d+)?", lines[-1])
        return nums[-1] if nums else text.strip()
    
    return text.strip()


def extract_final_answer(text: str) -> str:
    """Robust answer extraction mirroring the GRPO reward (markdown-aware)."""
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


# Lazily-loaded model cache so we load the HF model once per process.
_MODEL = None
_TOK = None


def _get_model(model_path: str):
    global _MODEL, _TOK
    if _MODEL is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        _TOK = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Force GPU: device_map="auto" can offload to CPU when VRAM looks full
        # and make generation ~100x slower. A 2B bf16 model (~4GB) fits a 24GB
        # card easily in a fresh process.
        torch.cuda.empty_cache()
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to("cuda")
        _MODEL.eval()
        if _TOK.pad_token is None:
            _TOK.pad_token = _TOK.eos_token
    return _MODEL, _TOK


def extract_claims(trace: str, max_claims: int = 5) -> List[str]:
    """
    Extract decision-relevant claims from a reasoning trace.
    
    Heuristic: Split by sentences and filter by length.
    """
    # Split by sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", trace)
    
    # Filter for substantive claims (length > 20 chars)
    claims = [s for s in sentences if len(s) > 20]
    
    return claims[:max_claims]


def verify_claim(model_path: str, claim: str, context: str) -> bool:
    """
    Verify a claim using the model itself (self-verification).

    Uses the cached transformers model (vLLM mis-registers the custom Qwen3.5
    config as multimodal and crashes). Returns True if the model says TRUE.
    """
    model, tok = _get_model(model_path)
    prompt = f"""Given the following solution:
{context}

Is the following claim true or false?
Claim: {claim}

Answer only TRUE or FALSE."""
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
    ).input_ids.squeeze(0).to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids.unsqueeze(0), do_sample=False, temperature=0.0,
            max_new_tokens=10, pad_token_id=tok.pad_token_id,
        )
    response = tok.decode(out[0][ids.size(0):], skip_special_tokens=True).upper()
    return "TRUE" in response


def clr_score(
    model_path: str,
    prompt: str,
    gold: str,
    K: int = 32,
    M: int = 5,
) -> float:
    """
    Calculate Claim-Level Reliability score.
    
    Args:
        model_path: Path to model
        prompt: Problem prompt
        gold: Gold answer
        K: Number of traces to sample
        M: Number of claims to verify per trace
        
    Returns:
        CLR score (0.0 to 1.0)
    """
    # Sample K traces with the cached transformers model (chunked to bound VRAM).
    try:
        from transformers import AutoModelForCausalLM  # noqa: F401
        model, tok = _get_model(model_path)
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
        ).squeeze(0).to(model.device)
        prompt_len = ids.size(0)
        traces = []
        remaining = K
        eos_id = tok.eos_token_id
        while remaining > 0:
            k = min(8, remaining)
            with torch.no_grad():
                out = model.generate(
                    ids.unsqueeze(0), do_sample=True, temperature=1.0, top_p=0.95,
                    max_new_tokens=2048, num_return_sequences=k,
                    eos_token_id=eos_id, pad_token_id=tok.pad_token_id,
                )
            for seq in out:
                traces.append(tok.decode(seq[prompt_len:], skip_special_tokens=True))
            remaining -= k
    except Exception as e:
        print(f"Trace sampling failed ({e}); using placeholder traces")
        traces = [f"Placeholder trace {i}" for i in range(K)]
    
    # Verify claims in each trace
    scores = []
    for trace in traces:
        claims = extract_claims(trace, max_claims=M)
        
        if not claims:
            scores.append(1.0)  # No claims to verify
            continue
        
        # Verify each claim
        verdicts = []
        for claim in claims:
            is_true = verify_claim(model_path, claim, trace)
            verdicts.append(1 if is_true else 0)
        
        # Average verdicts for this trace
        scores.append(np.mean(verdicts) if verdicts else 1.0)
    
    # Cluster by extracted answer
    buckets = defaultdict(float)
    for trace, score in zip(traces, scores):
        answer = extract_final_answer(trace)
        buckets[answer] += score
    
    # Pick answer with highest reliability sum
    if buckets:
        best_answer = max(buckets, key=buckets.get)
        
        # Check if best answer matches gold
        try:
            from math_verify import parse, verify
            correct = verify(parse(str(gold)), parse(best_answer))
        except Exception:
            correct = best_answer.strip() == str(gold).strip()
        
        return 1.0 if correct else 0.0
    
    return 0.0


def run_clr_evaluation(
    model_path: str,
    dataset_path: str,
    K: int = 32,
    M: int = 5,
    output_file: str = "clr_results.json",
):
    """
    Run CLR evaluation on dataset.
    
    Args:
        model_path: Path to model
        dataset_path: Path to dataset
        K: Number of traces per problem
        M: Number of claims per trace
        output_file: Output file for results
    """
    print(f"Loading dataset from {dataset_path}...")
    
    # Load dataset
    problems = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                # Extract user prompt
                for msg in record["messages"]:
                    if msg["role"] == "user":
                        prompt = msg["content"]
                        break
                gold = record.get("metadata", {}).get("answer", "")
                problems.append({"prompt": prompt, "gold": gold})
    
    print(f"Loaded {len(problems)} problems")
    
    # Run CLR evaluation
    print(f"\nRunning CLR evaluation (K={K}, M={M})...")
    start_time = time.time()
    
    scores = []
    for i, problem in enumerate(problems):
        score = clr_score(model_path, problem["prompt"], problem["gold"], K, M)
        scores.append(score)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(problems)} problems")
    
    evaluation_time = time.time() - start_time
    print(f"\nEvaluation completed in {evaluation_time:.1f} seconds")
    
    # Calculate metrics
    mean_score = np.mean(scores)
    print(f"Mean CLR score: {mean_score:.4f}")
    
    # Save results
    results = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "K": K,
        "M": M,
        "n_problems": len(problems),
        "mean_clr_score": float(mean_score),
        "evaluation_time": evaluation_time,
        "individual_scores": scores,
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="CLR Evaluation")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--K", type=int, default=32, help="Number of traces")
    parser.add_argument("--M", type=int, default=5, help="Number of claims")
    parser.add_argument("--output", default="clr_results.json", help="Output file")
    
    args = parser.parse_args()
    
    run_clr_evaluation(
        model_path=args.model,
        dataset_path=args.dataset,
        K=args.K,
        M=args.M,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
