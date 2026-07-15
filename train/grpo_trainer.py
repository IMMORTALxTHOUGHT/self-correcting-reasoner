#!/usr/bin/env python3
"""
GRPO Trainer with custom rewards for mathematical reasoning.

Uses TRL's GRPOTrainer with verifiable rewards (accuracy, format, self-correction).
"""

import argparse
import re
import time
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer


# ============================================================================
# Reward Functions
# ============================================================================

def format_reward(completions: List[dict], **kwargs) -> List[float]:
    """Reward proper <thinking>...</thinking><answer>...</answer> structure."""
    rewards = []
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
        score = 0.0
        if "<thinking>" in content and "</thinking>" in content:
            score += 0.4
        if "<answer>" in content and "</answer>" in content:
            score += 0.4
        if content.count("\n") >= 2:
            score += 0.2
        rewards.append(score)
    return rewards


def self_correction_reward(completions: List[dict], **kwargs) -> List[float]:
    """Reward genuine backtracking patterns in thinking blocks."""
    patterns = [
        r"wait,? let me",
        r"actually,? i (made|was) (a )?mistake",
        r"let me (rethink|recalculate|reconsider)",
        r"correction:",
        r"scratch that",
    ]
    rewards = []
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
        thinking = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
        if thinking:
            text = thinking.group(1)
            score = 0.5 if any(re.search(p, text, re.IGNORECASE) for p in patterns) else 0.0
            if "therefore" in text.lower():
                score += 0.3
            if text.count("\n") >= 3:
                score += 0.2
            rewards.append(min(score, 1.0))
        else:
            rewards.append(0.0)
    return rewards


def accuracy_reward(completions: List[dict], answers: List[str], **kwargs) -> List[float]:
    """Verifiable reward using math_verify."""
    rewards = []
    for comp, gold in zip(completions, answers):
        content = comp[0]["content"] if isinstance(comp, list) else comp
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if match:
            pred = match.group(1).strip()
            try:
                from math_verify import parse, verify
                ok = verify(parse(str(gold)), parse(pred))
            except Exception:
                ok = pred == str(gold).strip()
            rewards.append(1.0 if ok else 0.0)
        else:
            rewards.append(0.0)
    return rewards


def combined_reward(completions: List[dict], answers: List[str], **kwargs) -> List[float]:
    """Weighted combination of all reward signals."""
    fmt = format_reward(completions, **kwargs)
    corr = self_correction_reward(completions, **kwargs)
    acc = accuracy_reward(completions, answers=answers, **kwargs)
    return [0.3 * f + 0.3 * c + 0.4 * a for f, c, a in zip(fmt, corr, acc)]


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path: str, lora_r: int = 16, lora_alpha: int = 16):
    """Load model with QLoRA for GRPO training."""

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    return model, tokenizer


# ============================================================================
# Dataset
# ============================================================================

def prepare_dataset(dataset_path: str, tokenizer, max_seq_length: int = 2048):
    """Prepare dataset for GRPO: extract prompt + gold answer."""

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def process(example):
        user_prompt = next(m["content"] for m in example["messages"] if m["role"] == "user")
        gold = example.get("metadata", {}).get("answer", "")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "answer": gold}

    dataset = dataset.map(process)
    dataset = dataset.filter(lambda x: len(x["prompt"]) < max_seq_length // 2)
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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, tokenizer = load_model(model_path)
    dataset = prepare_dataset(dataset_path, tokenizer, max_seq_length)

    def reward_func(completions, **kwargs):
        return combined_reward(completions, answers=kwargs.get("answer", []))

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
        seed=42,
        max_seq_length=max_seq_length,
        num_generations=num_generations,
        beta=beta,
        temperature=1.0,
        top_p=0.95,
        report_to=["tensorboard"] + (["wandb"] if use_wandb else []),
        run_name="grpo-reasoner",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting GRPO training...")
    start = time.time()
    result = trainer.train()
    print(f"Done in {(time.time() - start) / 60:.1f} min")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/sft")
    parser.add_argument("--dataset", default="data/processed/gsm8k_train.jsonl")
    parser.add_argument("--output", default="runs/grpo")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
