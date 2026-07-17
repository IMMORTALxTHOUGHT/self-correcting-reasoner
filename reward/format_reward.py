#!/usr/bin/env python3
"""
Format Reward Function

Rewards proper format with <thinking> and <answer> tags.
"""

import re
from typing import List, Dict, Any


def format_reward(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Reward for proper format with thinking and answer tags.

    Scoring:
    - 0.4 for thinking tags
    - 0.4 for answer tags
    - 0.2 for proper structure

    Args:
        completions: List of completion dictionaries

    Returns:
        List of rewards between 0.0 and 1.0
    """
    rewards = []

    for comp in completions:
        # Handle different completion formats
        if isinstance(comp, list):
            content = comp[0]["content"]
        elif isinstance(comp, dict):
            content = comp.get("content", "")
        else:
            content = str(comp)

        score = 0.0

        # Check for thinking tags
        if "<thinking>" in content and "</thinking>" in content:
            score += 0.4

        # Check for answer tags
        if "<answer>" in content and "</answer>" in content:
            score += 0.4

        # Check for proper structure (multiple lines)
        if content.count("\n") >= 2:
            score += 0.2

        rewards.append(score)

    return rewards


def graded_format_reward(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Continuous format reward that yields *intra-group* variance.

    The binary format_reward saturates at 1.0 for almost every post-SFT
    completion, which collapses GRPO's group reward std to 0. This graded
    variant rewards partial / graded structure so different generations score
    differently (proper tag ordering, a non-empty <thinking> and <answer>,
    and answer length), giving the policy a real advantage signal.
    """
    rewards = []
    for comp in completions:
        if isinstance(comp, list):
            content = comp[0]["content"]
        elif isinstance(comp, dict):
            content = comp.get("content", "")
        else:
            content = str(comp)

        score = 0.0
        has_open_t = "<thinking>" in content
        has_close_t = "</thinking>" in content
        has_open_a = "<answer>" in content
        has_close_a = "</answer>" in content

        # Tags present (each worth 0.25), with a bonus for correct ordering.
        if has_open_t:
            score += 0.25
        if has_close_t:
            score += 0.25
        if has_open_a:
            score += 0.25
        if has_close_a:
            score += 0.25
        # Correct ordering: </thinking> must precede <answer>.
        if has_close_t and has_open_a and content.index("</thinking>") < content.index("<answer>"):
            score += 0.1

        # Non-empty thinking body (rewards elaboration, varies by generation).
        t_match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
        if t_match:
            t_len = len(t_match.group(1).strip())
            score += min(t_len / 200.0, 0.1)  # up to +0.1 for ~200 chars

        # Non-empty answer body.
        a_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if a_match:
            a_len = len(a_match.group(1).strip())
            score += min(a_len / 20.0, 0.1)  # up to +0.1 for ~20 chars

        rewards.append(min(score, 1.0))
    return rewards



def validate_format(text: str) -> Dict[str, Any]:
    """
    Validate format and return detailed analysis.
    
    Returns:
        Dictionary with validation results
    """
    has_thinking_open = "<thinking>" in text
    has_thinking_close = "</thinking>" in text
    has_answer_open = "<answer>" in text
    has_answer_close = "</answer>" in text
    
    # Extract thinking content
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    thinking_content = thinking_match.group(1) if thinking_match else ""
    
    # Extract answer content
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    answer_content = answer_match.group(1) if answer_match else ""
    
    # Check structure
    line_count = text.count("\n")
    
    return {
        "valid_format": all([
            has_thinking_open, has_thinking_close,
            has_answer_open, has_answer_close
        ]),
        "has_thinking_tags": has_thinking_open and has_thinking_close,
        "has_answer_tags": has_answer_open and has_answer_close,
        "thinking_content": thinking_content.strip(),
        "answer_content": answer_content.strip(),
        "line_count": line_count,
        "score": calculate_format_score(text),
    }


def calculate_format_score(text: str) -> float:
    """Calculate format score for a single text."""
    score = 0.0
    
    if "<thinking>" in text and "</thinking>" in text:
        score += 0.4
    
    if "<answer>" in text and "</answer>" in text:
        score += 0.4
    
    if text.count("\n") >= 2:
        score += 0.2
    
    return score


if __name__ == "__main__":
    # Test examples
    test_cases = [
        "<thinking>\nStep 1\nStep 2\n</thinking>\n<answer>\n42\n</answer>",
        "<thinking>\nStep 1\n</thinking>\n<answer>42</answer>",
        "Just a plain answer: 42",
        "<thinking>\nCorrect format\n</thinking>",
    ]
    
    for i, text in enumerate(test_cases):
        print(f"\nTest case {i + 1}:")
        print(f"Text: {text[:50]}...")
        print(f"Score: {calculate_format_score(text)}")
        validation = validate_format(text)
        print(f"Valid format: {validation['valid_format']}")
