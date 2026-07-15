#!/usr/bin/env python3
"""
GSM8K Dataset Preparation for Self-Correcting Reasoner

Downloads and processes GSM8K dataset into training format with
reasoning traces in <thinking>...</thinking><answer>...</answer> structure.
"""

import json
import re
from pathlib import Path
from datasets import load_dataset


def extract_answer(answer_text: str) -> str:
    """Extract numerical answer from GSM8K answer text."""
    # GSM8K answers end with #### <number>
    match = re.search(r"####\s*(.+)$", answer_text)
    if match:
        return match.group(1).strip()
    return answer_text.strip()


def create_reasoning_trace(question: str, solution: str) -> str:
    """
    Create structured reasoning trace with thinking and answer blocks.
    
    Template:
    <thinking>
    Step-by-step reasoning...
    </thinking>
    <answer>
    final answer
    </answer>
    """
    answer = extract_answer(solution)
    
    # Clean up solution text (remove #### part)
    reasoning = re.sub(r"####\s*.+$", "", solution, flags=re.MULTILINE).strip()
    
    trace = f"""<thinking>
{reasoning}
</thinking>
<answer>
{answer}
</answer>"""
    
    return trace


def prepare_gsm8k(output_dir: str = "data/processed"):
    """Download and prepare GSM8K dataset."""
    
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name in ["train", "test"]:
        print(f"Processing {split_name} split...")
        split = dataset[split_name]
        
        records = []
        for item in split:
            question = item["question"]
            solution = item["answer"]
            
            # Create reasoning trace
            assistant_content = create_reasoning_trace(question, solution)
            
            record = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": assistant_content}
                ],
                "metadata": {
                    "dataset": "gsm8k",
                    "split": split_name,
                    "answer": extract_answer(solution)
                }
            }
            records.append(record)
        
        # Save as JSONL
        output_file = output_path / f"gsm8k_{split_name}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"Saved {len(records)} records to {output_file}")
    
    return dataset


if __name__ == "__main__":
    prepare_gsm8k()
