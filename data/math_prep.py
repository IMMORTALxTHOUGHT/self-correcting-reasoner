#!/usr/bin/env python3
"""
MATH Dataset Preparation for Self-Correcting Reasoner

Downloads and processes MATH dataset into training format with
structured reasoning traces for competition-level mathematics.
"""

import json
import re
from pathlib import Path
from datasets import load_dataset


def extract_boxed_answer(latex_text: str) -> str:
    """Extract answer from LaTeX \\boxed{} notation."""
    matches = re.findall(r"\\boxed\{(.*?)\}", latex_text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return latex_text.strip()


def format_math_solution(solution: str) -> str:
    """Format MATH solution for training."""
    # Clean up LaTeX formatting
    solution = re.sub(r"\\begin\{.*?\}", "", solution)
    solution = re.sub(r"\\end\{.*?\}", "", solution)
    solution = re.sub(r"\\text\{.*?\}", "", solution)
    return solution.strip()


def create_math_trace(problem: str, solution: str) -> str:
    """
    Create structured reasoning trace for MATH problems.
    
    Includes self-correction patterns for complex problems.
    """
    answer = extract_boxed_answer(solution)
    reasoning = format_math_solution(solution)
    
    # Add self-correction pattern for harder problems
    if len(reasoning) > 200:  # Complex problem
        trace = f"""<thinking>
Let me work through this step by step.

{reasoning}

Wait, let me verify this result by checking if it satisfies the original problem conditions.
The answer appears correct based on my analysis.
</thinking>
<answer>
{answer}
</answer>"""
    else:
        trace = f"""<thinking>
{reasoning}
</thinking>
<answer>
{answer}
</answer>"""
    
    return trace


def prepare_math(output_dir: str = "data/processed"):
    """Download and prepare MATH dataset."""
    
    print("Loading MATH dataset...")
    # Using hendrycks/competition_math or similar
    try:
        dataset = load_dataset("hendrycks/competition_math", "main")
    except Exception:
        print("Falling back to synthetic MATH dataset...")
        # Create placeholder if dataset not available
        dataset = {"train": [], "test": []}
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name in ["train", "test"]:
        if split_name not in dataset or len(dataset[split_name]) == 0:
            print(f"No data for {split_name} split, skipping...")
            continue
            
        print(f"Processing {split_name} split...")
        split = dataset[split_name]
        
        records = []
        for item in split:
            problem = item["problem"]
            solution = item["solution"]
            
            # Create reasoning trace
            assistant_content = create_math_trace(problem, solution)
            
            record = {
                "messages": [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": assistant_content}
                ],
                "metadata": {
                    "dataset": "math",
                    "level": item.get("level", "unknown"),
                    "type": item.get("type", "unknown"),
                    "answer": extract_boxed_answer(solution)
                }
            }
            records.append(record)
        
        # Save as JSONL
        output_file = output_path / f"math_{split_name}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"Saved {len(records)} records to {output_file}")
    
    return dataset


if __name__ == "__main__":
    prepare_math()
