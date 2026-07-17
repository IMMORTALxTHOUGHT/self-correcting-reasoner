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
    Continuous format reward that yields *intra-group* variance AND works even
    when the model does not emit the <thinking>/<answer> tags.

    The tags are treated as *bonuses* (correct structure is encouraged) rather
    than hard requirements: if absent, the model still scores from its
    reasoning text. This avoids the all-zero reward that occurs when a model
    answers in plain markdown and the previous binary reward found no tags.
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

        # Tag presence bonuses (each worth 0.2), plus ordering bonus.
        tag_score = 0.0
        tag_score += 0.2 if has_open_t else 0.0
        tag_score += 0.2 if has_close_t else 0.0
        tag_score += 0.2 if has_open_a else 0.0
        tag_score += 0.2 if has_close_a else 0.0
        if has_close_t and has_open_a and content.index("</thinking>") < content.index("<answer>"):
            tag_score += 0.1
        score += tag_score

        # Always reward substance so untagged (markdown) outputs still score:
        # non-empty reasoning body + a concrete final answer candidate.
        t_len = len(content.strip())
        score += min(t_len / 400.0, 0.2)  # up to +0.2 for ~400 chars of reasoning

        a = _extract_final_answer(content)
        if a:
            score += 0.2  # produced an extractable final answer

        rewards.append(min(score, 1.0))
    return rewards


def _extract_final_answer(text: str) -> str:
    """Robustly pull the model's final answer out of arbitrary output.

    Tries, in order: <answer> tags, \\boxed{}, a **bold** final value, the last
    dollar/number on the last line, then the last number in the text. Returns
    '' if nothing numeric is found. Strips $, commas, and surrounding markdown.
    """
    if not text:
        return ""

    # 1) <answer> ... </answer>
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2) \boxed{...}
    m = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 3) Last **bold** token (markdown final answer), e.g. "**$96**"
    bolds = re.findall(r"\*\*(.+?)\*\*", text)
    if bolds:
        last = bolds[-1].strip()
        if re.search(r"\d", last):
            return last

    # 4) Last line: a trailing $number or bare number
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        nums = re.findall(r"\$?-?[\d,]+(?:\.\d+)?", lines[-1])
        if nums:
            return nums[-1].replace(",", "")

    # 5) Last number anywhere
    all_nums = re.findall(r"\$?-?[\d,]+(?:\.\d+)?", text)
    if all_nums:
        return all_nums[-1].replace(",", "")

    return ""



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
