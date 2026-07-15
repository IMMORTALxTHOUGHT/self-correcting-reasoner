#!/usr/bin/env python3
"""
GRPO Training Configuration

Configuration for Group Relative Policy Optimization alignment.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    
    # Model settings
    model_path: str = "runs/sft"  # Path to SFT checkpoint
    max_seq_length: int = 2048
    
    # GRPO specific settings
    num_generations: int = 8  # Group size for baseline estimation
    beta: float = 0.1  # KL penalty coefficient
    temperature: float = 1.0  # Sampling temperature
    top_p: float = 0.95
    top_k: int = -1  # No top-k filtering
    
    # Training hyperparameters
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5  # Conservative for stability
    num_train_epochs: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Optimization
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "linear"
    fp16: bool = False
    bf16: bool = True
    
    # Checkpointing
    output_dir: str = "runs/grpo"
    logging_steps: int = 10
    save_steps: int = 50
    save_total_limit: int = 3
    
    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 100
    
    # Reproducibility
    seed: int = 42
    
    # Dataset
    dataset_path: str = "data/processed/gsm8k_train.jsonl"
    eval_dataset_path: Optional[str] = "data/processed/gsm8k_test.jsonl"
    
    # Reward weights
    reward_weights: dict = field(default_factory=lambda: {
        "format": 0.3,
        "self_correction": 0.3,
        "accuracy": 0.4,
    })
    
    # Logging
    report_to: list = field(default_factory=lambda: ["tensorboard", "wandb"])
    run_name: str = "grpo-reasoner"


@dataclass
class RewardConfig:
    """Configuration for reward functions."""
    
    # Format reward settings
    format_reward_weight: float = 0.3
    thinking_tag_bonus: float = 0.4
    answer_tag_bonus: float = 0.4
    structure_bonus: float = 0.2
    
    # Self-correction reward settings
    self_correction_weight: float = 0.3
    backtracking_patterns: list = field(default_factory=lambda: [
        r"wait,? let me",
        r"actually,? i (made|was) (a )?mistake",
        r"let me (rethink|recalculate|reconsider)",
        r"i (was|made) wrong",
        r"correction:",
        r"scratch that",
    ])
    structured_correction_bonus: float = 0.3
    multiple_steps_bonus: float = 0.2
    
    # Accuracy reward settings
    accuracy_weight: float = 0.4
    
    # Reward clipping
    min_reward: float = 0.0
    max_reward: float = 1.0


def get_grpo_config() -> GRPOConfig:
    """Get default GRPO configuration."""
    return GRPOConfig()


def get_reward_config() -> RewardConfig:
    """Get default reward configuration."""
    return RewardConfig()


if __name__ == "__main__":
    config = get_grpo_config()
    print("GRPO Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
