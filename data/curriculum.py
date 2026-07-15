#!/usr/bin/env python3
"""
Curriculum Learning Strategy

Stratifies training data by difficulty for progressive learning:
1. Stage 1: Simple arithmetic (easy)
2. Stage 2: Multi-step problems (medium)
3. Stage 3: Competition math (hard)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple


def estimate_difficulty(record: Dict[str, Any]) -> str:
    """
    Estimate problem difficulty based on heuristics.
    
    Returns: 'easy', 'medium', or 'hard'
    """
    question = record["messages"][0]["content"]
    answer_content = record["messages"][1]["content"]
    
    # Extract thinking content
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", answer_content, re.DOTALL)
    thinking = thinking_match.group(1) if thinking_match else ""
    
    # Heuristics for difficulty
    score = 0
    
    # Question length
    if len(question) > 200:
        score += 2
    elif len(question) > 100:
        score += 1
    
    # Number of steps in thinking
    step_count = len([line for line in thinking.split("\n") if line.strip()])
    if step_count > 10:
        score += 2
    elif step_count > 5:
        score += 1
    
    # Presence of complex operations
    complex_patterns = [
        r"integral", r"derivative", r"matrix", r"polynomial",
        r"quadratic", r"simultaneous", r"inequality"
    ]
    for pattern in complex_patterns:
        if re.search(pattern, thinking, re.IGNORECASE):
            score += 1
    
    # Self-correction indicates complexity
    correction_patterns = [
        r"wait,? let me", r"actually", r"correction", r"scratch that"
    ]
    for pattern in correction_patterns:
        if re.search(pattern, thinking, re.IGNORECASE):
            score += 1
    
    # Classify based on score
    if score <= 2:
        return "easy"
    elif score <= 4:
        return "medium"
    else:
        return "hard"


def stratify_dataset(
    input_file: str,
    output_dir: str = "data/curriculum"
) -> Dict[str, int]:
    """
    Stratify dataset into curriculum stages.
    
    Returns: Dictionary with counts for each stage
    """
    print(f"Loading dataset from {input_file}...")
    
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} records")
    
    # Categorize by difficulty
    stages = {"easy": [], "medium": [], "hard": []}
    
    for record in records:
        difficulty = estimate_difficulty(record)
        stages[difficulty].append(record)
    
    # Print statistics
    print("\nDifficulty distribution:")
    for stage, items in stages.items():
        print(f"  {stage}: {len(items)} records")
    
    # Save stratified datasets
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    counts = {}
    for stage, items in stages.items():
        output_file = output_path / f"{stage}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        counts[stage] = len(items)
        print(f"Saved {len(items)} records to {output_file}")
    
    # Create combined curriculum file
    curriculum_config = {
        "stages": [
            {"name": "stage1_easy", "file": "easy.jsonl", "epochs": 3},
            {"name": "stage2_medium", "file": "medium.jsonl", "epochs": 2},
            {"name": "stage3_hard", "file": "hard.jsonl", "epochs": 1},
        ],
        "total_records": len(records)
    }
    
    config_file = output_path / "curriculum_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(curriculum_config, f, indent=2)
    
    print(f"\nCurriculum config saved to {config_file}")
    
    return counts


def create_curriculum_datalist(
    curriculum_dir: str = "data/curriculum"
) -> List[str]:
    """
    Create ordered list of dataset files for curriculum training.
    
    Returns: List of file paths in training order
    """
    curriculum_path = Path(curriculum_dir)
    
    # Load curriculum config
    config_file = curriculum_path / "curriculum_config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        
        return [curriculum_path / stage["file"] for stage in config["stages"]]
    
    # Default order if no config exists
    return [
        curriculum_path / "easy.jsonl",
        curriculum_path / "medium.jsonl",
        curriculum_path / "hard.jsonl",
    ]


if __name__ == "__main__":
    # Example: stratify GSM8K training data
    input_file = "data/processed/all_train.jsonl"

    if Path(input_file).exists():
        counts = stratify_dataset(input_file)
        print(f"\nStratification complete: {counts}")
    else:
        print(f"Input file not found: {input_file}")
        print("Run python3 data/prepare_all.py first to generate the dataset.")
