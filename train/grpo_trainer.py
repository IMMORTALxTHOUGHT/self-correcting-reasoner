#!/usr/bin/env python3
"""
GRPO Trainer with Custom Rewards

Implements Group Relative Policy Optimization for alignment training
with custom rewards for format, self-correction, and accuracy.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import GRPOConfig, GRPOTrainer


# ============================================================================
# Reward Functions
# ============================================================================

def format_reward(completions: List[Dict], **kwargs) -> List[float]:
    """
    Reward for proper format with thinking and answer tags.
    
    Returns scores between 0.0 and 1.0:
    - 0.4 for thinking tags
    - 0.4 for answer tags
    - 0.2 for proper structure
    """
    rewards = []
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
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


def self_correction_reward(completions: List[Dict], **kwargs) -> List[float]:
    """
    Reward for genuine self-correction in reasoning.
    
    Detects backtracking patterns and rewards structured corrections.
    """
    backtracking_patterns = [
        r"wait,? let me",
        r"actually,? i (made|was) (a )?mistake",
        r"let me (rethink|recalculate|reconsider)",
        r"i (was|made) wrong",
        r"correction:",
        r"scratch that",
    ]
    
    rewards = []
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
        
        # Extract thinking block
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
        
        if thinking_match:
            thinking = thinking_match.group(1)
            
            # Check for backtracking patterns
            has_backtracking = any(
                re.search(pattern, thinking, re.IGNORECASE)
                for pattern in backtracking_patterns
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


def accuracy_reward(completions: List[Dict], answers: List[str], **kwargs) -> List[float]:
    """
    Reward for correct final answer.
    
    Uses math_verify for exact answer matching.
    """
    rewards = []
    for comp, gold in zip(completions, answers):
        content = comp[0]["content"] if isinstance(comp, list) else comp
        
        # Extract answer from tags
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        
        if answer_match:
            pred = answer_match.group(1).strip()
            
            # Use math_verify for exact matching
            try:
                from math_verify import parse, verify
                ok = verify(parse(str(gold)), parse(pred))
            except Exception:
                # Fallback to string comparison
                ok = pred.strip() == str(gold).strip()
            
            rewards.append(1.0 if ok else 0.0)
        else:
            rewards.append(0.0)
    
    return rewards


def combined_reward(
    completions: List[Dict],
    answers: List[str],
    weights: Optional[Dict[str, float]] = None,
    **kwargs
) -> List[float]:
    """
    Combined reward with weighted components.
    
    Args:
        completions: Model completions
        answers: Gold answers
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


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_grpo_dataset(
    dataset_path: str,
    tokenizer,
    max_seq_length: int = 2048,
):
    """Prepare dataset for GRPO training."""
    
    print(f"Loading dataset from {dataset_path}...")
    
    # Load JSONL dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # Extract prompts and answers
    def process_example(example):
        messages = example["messages"]
        
        # Get user prompt
        user_prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                user_prompt = msg["content"]
                break
        
        # Get gold answer from metadata
        gold_answer = example.get("metadata", {}).get("answer", "")
        
        # Apply chat template to prompt
        prompt_messages = [{"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        return {
            "prompt": prompt,
            "answer": gold_answer,
        }
    
    dataset = dataset.map(process_example)
    
    # Limit sequence length
    dataset = dataset.filter(lambda x: len(x["prompt"]) < max_seq_length // 2)
    
    print(f"Prepared {len(dataset)} examples for GRPO training")
    return dataset


# ============================================================================
# Training
# ============================================================================

def train(
    model_path: str = "runs/sft",
    dataset_path: str = "data/processed/gsm8k_train.jsonl",
    output_dir: str = "runs/grpo",
    num_generations: int = 8,
    beta: float = 0.1,
    learning_rate: float = 1e-5,
    num_epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_seq_length: int = 2048,
    use_wandb: bool = False,
):
    """Run GRPO training."""
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Load model
    try:
        from unsloth import FastLanguageModel
        
        print(f"Loading model from {model_path} with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        
        # Add LoRA if not already present
        if not hasattr(model, "peft_config"):
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                lora_alpha=16,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
        
        backend = "unsloth"
        
    except ImportError:
        print("Unsloth not available, using standard PEFT...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        backend = "peft"
    
    print(f"Model loaded via {backend}")
    
    # Prepare dataset
    train_dataset = prepare_grpo_dataset(dataset_path, tokenizer, max_seq_length)
    
    # Create reward functions with weights
    def reward_func(completions, **kwargs):
        return combined_reward(
            completions,
            answers=kwargs.get("answer", []),
            weights={
                "format": 0.3,
                "self_correction": 0.3,
                "accuracy": 0.4,
            }
        )
    
    # Training arguments
    training_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=100,
        seed=42,
        max_seq_length=max_seq_length,
        num_generations=num_generations,
        beta=beta,
        temperature=1.0,
        top_p=0.95,
        report_to=["tensorboard"] + (["wandb"] if use_wandb else []),
        run_name="grpo-reasoner",
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Train
    print("\nStarting GRPO training...")
    start_time = time.time()
    
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time / 60:.1f} minutes")
    
    # Print metrics
    if train_result.metrics:
        print(f"Final reward: {train_result.metrics.get('reward', 'N/A')}")
    
    # Save model
    print("\nSaving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nModel saved to {output_dir}")
    return train_result


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Self-Correcting Reasoner")
    parser.add_argument("--model", default="runs/sft", help="Path to SFT checkpoint")
    parser.add_argument("--dataset", default="data/processed/gsm8k_train.jsonl", help="Dataset path")
    parser.add_argument("--output", default="runs/grpo", help="Output directory")
    parser.add_argument("--num-generations", type=int, default=8, help="Group size")
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    
    args = parser.parse_args()
    
    train(
        model_path=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        num_generations=args.num_generations,
        beta=args.beta,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_length=args.max_seq_length,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
