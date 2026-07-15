#!/usr/bin/env python3
"""
Combined Reward Function

Combines format, self-correction, and accuracy rewards with configurable weights.
"""

import re
from typing import List, Dict, Any, Optional

from .format_reward import format_reward, calculate_format_score
from .self_correction_reward import self_correction_reward, calculate_correction_score
from .accuracy_reward import accuracy_reward, normalize_answer


def combined_reward(
    completions: List[Dict[str, Any]],
    answers: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    **kwargs
) -> List[float]:
    """
    Combined reward with weighted components.
    
    Args:
        completions: Model completions
        answers: Gold answers (required for accuracy reward)
        weights: Dictionary of reward weights
        
    Returns:
        Combined rewards between 0.0 and 1.0
    """
    if weights is None:
        weights = {
            "format": 0.3,
            "self_correction": 0.3,
            "accuracy": 0.4,
        }
    
    if answers is None:
        # If no answers provided, only use format and self-correction
        weights = {
            "format": 0.5,
            "self_correction": 0.5,
            "accuracy": 0.0,
        }
        answers = [""] * len(completions)
    
    format_rewards = format_reward(completions, **kwargs)
    correction_rewards = self_correction_reward(completions, **kwargs)
    accuracy_rewards = accuracy_reward(completions, answers, **kwargs)
    
    combined = []
    for f, c, a in zip(format_rewards, correction_rewards, accuracy_rewards):
        score = (
            weights["format"] * f +
            weights["self_correction"] * c +
            weights["accuracy"] * a
        )
        combined.append(score)
    
    return combined


def combined_reward_detailed(
    completions: List[Dict[str, Any]],
    answers: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Combined reward with detailed breakdown.
    
    Returns:
        List of dictionaries with reward details
    """
    if weights is None:
        weights = {
            "format": 0.3,
            "self_correction": 0.3,
            "accuracy": 0.4,
        }
    
    if answers is None:
        answers = [""] * len(completions)
    
    results = []
    for comp, gold in zip(completions, answers):
        # Handle different completion formats
        if isinstance(comp, list):
            content = comp[0]["content"]
        elif isinstance(comp, dict):
            content = comp.get("content", "")
        else:
            content = str(comp)
        
        # Calculate individual rewards
        format_score = calculate_format_score(content)
        correction_score = calculate_correction_score(content)
        
        # Accuracy score
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if answer_match and gold:
            pred = answer_match.group(1).strip()
            try:
                from math_verify import parse, verify
                accuracy_score = 1.0 if verify(parse(str(gold)), parse(pred)) else 0.0
            except Exception:
                accuracy_score = 1.0 if normalize_answer(pred) == normalize_answer(str(gold)) else 0.0
        else:
            accuracy_score = 0.0
        
        # Combined score
        combined_score = (
            weights["format"] * format_score +
            weights["self_correction"] * correction_score +
            weights["accuracy"] * accuracy_score
        )
        
        results.append({
            "combined_score": combined_score,
            "format_score": format_score,
            "self_correction_score": correction_score,
            "accuracy_score": accuracy_score,
            "weights": weights,
            "predicted_answer": answer_match.group(1).strip() if answer_match else None,
            "gold_answer": gold,
        })
    
    return results


if __name__ == "__main__":
    # Test examples
    test_cases = [
        {
            "content": "<thinking>\nLet me calculate 2 + 3 * 4.\nFirst, multiplication: 3 * 4 = 12.\nThen addition: 2 + 12 = 14.\n</thinking>\n<answer>\n14\n</answer>",
            "answer": "14",
        },
        {
            "content": "<thinking>\nWait, let me recalculate.\nActually, I made a mistake.\nThe correct answer is 42.\n</thinking>\n<answer>\n42\n</answer>",
            "answer": "42",
        },
        {
            "content": "Just a plain answer: 42",
            "answer": "42",
        },
    ]
    
    completions = [[{"content": tc["content"]}] for tc in test_cases]
    answers = [tc["answer"] for tc in test_cases]
    
    # Get combined rewards
    rewards = combined_reward(completions, answers)
    print("Combined rewards:", rewards)
    
    # Get detailed breakdown
    detailed = combined_reward_detailed(completions, answers)
    for i, result in enumerate(detailed):
        print(f"\nTest case {i + 1}:")
        print(f"  Combined: {result['combined_score']:.3f}")
        print(f"  Format: {result['format_score']:.3f}")
        print(f"  Self-correction: {result['self_correction_score']:.3f}")
        print(f"  Accuracy: {result['accuracy_score']:.3f}")
