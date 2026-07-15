#!/usr/bin/env python3
"""
Accuracy Reward Function

Rewards correct final answers using math_verify for exact matching.
"""

import re
from typing import List, Dict, Any


def accuracy_reward(completions: List[Dict[str, Any]], answers: List[str], **kwargs) -> List[float]:
    """
    Reward for correct final answer.
    
    Uses math_verify for exact answer matching when available,
    falls back to string comparison.
    
    Args:
        completions: List of completion dictionaries
        answers: List of gold answers
        
    Returns:
        List of rewards (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []
    
    for comp, gold in zip(completions, answers):
        # Handle different completion formats
        if isinstance(comp, list):
            content = comp[0]["content"]
        elif isinstance(comp, dict):
            content = comp.get("content", "")
        else:
            content = str(comp)
        
        # Extract answer from tags
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        
        if answer_match:
            pred = answer_match.group(1).strip()
            
            # Use math_verify for exact matching
            try:
                from math_verify import parse, verify
                ok = verify(parse(str(gold)), parse(pred))
            except ImportError:
                # Fallback to string comparison
                ok = normalize_answer(pred) == normalize_answer(str(gold))
            except Exception:
                # Fallback to string comparison
                ok = normalize_answer(pred) == normalize_answer(str(gold))
            
            rewards.append(1.0 if ok else 0.0)
        else:
            rewards.append(0.0)
    
    return rewards


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    
    Handles:
    - Whitespace normalization
    - LaTeX formatting
    - Common answer formats
    """
    # Remove LaTeX formatting
    answer = re.sub(r"\\boxed\{(.*?)\}", r"\1", answer)
    answer = re.sub(r"\\text\{(.*?)\}", r"\1", answer)
    answer = re.sub(r"\\frac\{(.*?)\}\{(.*?)\}", r"(\1)/(\2)", answer)
    
    # Remove extra whitespace
    answer = " ".join(answer.split())
    
    # Remove trailing periods
    answer = answer.rstrip(".")
    
    return answer.strip()


def extract_answer(text: str) -> str:
    """
    Extract answer from text.
    
    Tries multiple extraction methods:
    1. <answer> tags
    2. \\boxed{} notation
    3. Last line
    """
    # Try answer tags first
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Try \boxed{} notation
    boxed_match = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Fall back to last line
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        return lines[-1]
    
    return ""


def verify_answer(pred: str, gold: str) -> Dict[str, Any]:
    """
    Verify answer and return detailed results.
    
    Returns:
        Dictionary with verification results
    """
    normalized_pred = normalize_answer(pred)
    normalized_gold = normalize_answer(gold)
    
    # Try math_verify
    try:
        from math_verify import parse, verify
        math_ok = verify(parse(str(gold)), parse(pred))
    except Exception:
        math_ok = None
    
    # String comparison
    string_ok = normalized_pred == normalized_gold
    
    return {
        "correct": math_ok if math_ok is not None else string_ok,
        "math_verify_result": math_ok,
        "string_match_result": string_ok,
        "predicted": pred,
        "normalized_predicted": normalized_pred,
        "gold": gold,
        "normalized_gold": normalized_gold,
    }


if __name__ == "__main__":
    # Test examples
    test_cases = [
        ("42", "42"),
        ("14", "14"),
        ("$14$", "14"),
        ("\\boxed{100}", "100"),
        ("The answer is 42.", "42"),
        ("15", "14"),
    ]
    
    for pred, gold in test_cases:
        print(f"\nPredicted: {pred}")
        print(f"Gold: {gold}")
        result = verify_answer(pred, gold)
        print(f"Correct: {result['correct']}")
        print(f"Normalized pred: {result['normalized_predicted']}")
