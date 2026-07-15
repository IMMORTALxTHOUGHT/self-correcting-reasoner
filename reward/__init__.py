"""Reusable reward functions for GRPO training and evaluation.

Single source of truth for the format, self-correction, accuracy, and combined
reward signals. Import these instead of re-implementing them in trainers/evals.
"""

from .format_reward import (
    format_reward,
    calculate_format_score,
    validate_format,
)
from .self_correction_reward import (
    self_correction_reward,
    calculate_correction_score,
    detect_backtracking,
    BACKTRACKING_PATTERNS,
)
from .accuracy_reward import (
    accuracy_reward,
    normalize_answer,
    verify_answer,
)
from .combined_reward import (
    combined_reward,
    combined_reward_detailed,
)

__all__ = [
    "format_reward",
    "calculate_format_score",
    "validate_format",
    "self_correction_reward",
    "calculate_correction_score",
    "detect_backtracking",
    "BACKTRACKING_PATTERNS",
    "accuracy_reward",
    "normalize_answer",
    "verify_answer",
    "combined_reward",
    "combined_reward_detailed",
]
