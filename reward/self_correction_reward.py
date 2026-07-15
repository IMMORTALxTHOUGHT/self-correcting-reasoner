#!/usr/bin/env python3
"""
Self-Correction Reward Function

Rewards genuine self-correction and backtracking in reasoning.
"""

import re
from typing import List, Dict, Any


# Patterns indicating self-correction
BACKTRACKING_PATTERNS = [
    r"wait,? let me",
    r"actually,? i (made|was) (a )?mistake",
    r"let me (rethink|recalculate|reconsider)",
    r"i (was|made) wrong",
    r"correction:",
    r"scratch that",
    r"hold on",
    r"let me check",
    r"i need to (fix|correct|update)",
    r"that's not right",
    r"let me try again",
]


def self_correction_reward(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Reward for genuine self-correction in reasoning.
    
    Scoring:
    - 0.5 for backtracking patterns
    - 0.3 for structured correction (arrows, therefore)
    - 0.2 for multiple reasoning steps
    
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
        
        # Extract thinking block
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
        
        if thinking_match:
            thinking = thinking_match.group(1)
            
            # Check for backtracking patterns
            has_backtracking = any(
                re.search(pattern, thinking, re.IGNORECASE)
                for pattern in BACKTRACKING_PATTERNS
            )
            score = 0.5 if has_backtracking else 0.0
            
            # Bonus for structured correction
            if "→" in thinking or "therefore" in thinking.lower():
                score += 0.3
            
            # Bonus for multiple reasoning steps
            if thinking.count("\n") >= 3:
                score += 0.2
            
            rewards.append(min(score, 1.0))
        else:
            rewards.append(0.0)
    
    return rewards


def detect_backtracking(text: str) -> Dict[str, Any]:
    """
    Detect and analyze backtracking in text.
    
    Returns:
        Dictionary with backtracking analysis
    """
    # Extract thinking content
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    thinking = thinking_match.group(1) if thinking_match else text
    
    # Find all backtracking instances
    backtracking_instances = []
    for pattern in BACKTRACKING_PATTERNS:
        matches = re.finditer(pattern, thinking, re.IGNORECASE)
        for match in matches:
            backtracking_instances.append({
                "pattern": pattern,
                "match": match.group(),
                "start": match.start(),
                "end": match.end(),
            })
    
    # Check for structured correction
    has_arrow = "→" in thinking
    has_therefore = "therefore" in thinking.lower()
    
    # Count reasoning steps
    line_count = len([line for line in thinking.split("\n") if line.strip()])
    
    return {
        "has_backtracking": len(backtracking_instances) > 0,
        "backtracking_count": len(backtracking_instances),
        "backtracking_instances": backtracking_instances,
        "has_structured_correction": has_arrow or has_therefore,
        "reasoning_steps": line_count,
        "score": calculate_correction_score(text),
    }


def calculate_correction_score(text: str) -> float:
    """Calculate self-correction score for a single text."""
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    
    if not thinking_match:
        return 0.0
    
    thinking = thinking_match.group(1)
    
    # Check for backtracking
    has_backtracking = any(
        re.search(pattern, thinking, re.IGNORECASE)
        for pattern in BACKTRACKING_PATTERNS
    )
    score = 0.5 if has_backtracking else 0.0
    
    # Bonus for structured correction
    if "→" in thinking or "therefore" in thinking.lower():
        score += 0.3
    
    # Bonus for multiple reasoning steps
    if thinking.count("\n") >= 3:
        score += 0.2
    
    return min(score, 1.0)


if __name__ == "__main__":
    # Test examples
    test_cases = [
        """<thinking>
Let me calculate 2 + 3 * 4.
First, multiplication: 3 * 4 = 12.
Then addition: 2 + 12 = 14.
</thinking>
<answer>14</answer>""",
        
        """<thinking>
I need to solve this equation.
Wait, let me recalculate.
Actually, I made a mistake in step 2.
The correct answer is 42.
</thinking>
<answer>42</answer>""",
        
        """<thinking>
Step 1: Identify the problem.
Step 2: Set up the equation.
Step 3: Solve step by step.
Therefore, the answer is 100.
</thinking>
<answer>100</answer>""",
    ]
    
    for i, text in enumerate(test_cases):
        print(f"\nTest case {i + 1}:")
        print(f"Score: {calculate_correction_score(text)}")
        analysis = detect_backtracking(text)
        print(f"Has backtracking: {analysis['has_backtracking']}")
        print(f"Backtracking count: {analysis['backtracking_count']}")
