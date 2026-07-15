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
