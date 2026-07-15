#!/usr/bin/env python3
"""
SFT Training Configuration

Configuration for Supervised Fine-Tuning with QLoRA on Qwen 3.5 4B.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    
    # Model settings
    base_model: str = "Qwen/Qwen3-4B"  # Fallback: "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 2048
    
    # QLoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training hyperparameters
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    
    # Optimization
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "linear"
    fp16: bool = False
    bf16: bool = True
    
    # Checkpointing
    output_dir: str = "runs/sft"
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    
    # Evaluation
    eval_strategy: str = "epoch"
    eval_steps: Optional[int] = None
    
    # Reproducibility
    seed: int = 42
    
    # Dataset
    dataset_path: str = "data/processed/gsm8k_train.jsonl"
    eval_dataset_path: Optional[str] = "data/processed/gsm8k_test.jsonl"
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: list = field(default_factory=lambda: [
        {"name": "easy", "epochs": 3},
        {"name": "medium", "epochs": 2},
        {"name": "hard", "epochs": 1},
    ])
    
    # Logging
    report_to: list = field(default_factory=lambda: ["tensorboard", "wandb"])
    run_name: str = "sft-reasoner"


@dataclass
class QLoRAConfig:
    """QLoRA-specific configuration."""
    
    # Quantization settings
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    
    # LoRA settings
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    # Gradient checkpointing
    use_gradient_checkpointing: str = "unsloth"
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {
        "use_reentrant": False
    })


def get_sft_config() -> SFTConfig:
    """Get default SFT configuration."""
    return SFTConfig()


def get_qlora_config() -> QLoRAConfig:
    """Get default QLoRA configuration."""
    return QLoRAConfig()


if __name__ == "__main__":
    config = get_sft_config()
    print("SFT Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
