#!/usr/bin/env python3
"""
GRPO Trainer with custom rewards for mathematical reasoning.

Uses TRL's GRPOTrainer with verifiable rewards (accuracy, format, self-correction).
"""

import argparse
import os
import re
import sys
import time
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer

# Make the project root importable when run as `python train/grpo_trainer.py`
# (sys.path[0] is the train/ dir, not the project root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# Reward Functions
# ============================================================================
# The reward logic lives in the reusable `reward/` package so there is a single
# source of truth (previously it was duplicated inline here and drifted).
from reward.combined_reward import combined_reward  # noqa: E402


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
    dataset_path: str = "data/processed/all_train.jsonl",
    output_dir: str = "runs/grpo",
    num_generations: int = 8,
    beta: float = 0.1,
    learning_rate: float = 1e-5,
    num_epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_seq_length: int = 2048,
    max_steps: Optional[int] = None,
    vllm_gpu_memory_utilization: float = 0.3,
    use_wandb: bool = False,
    vllm_kwargs: Optional[dict] = None,
):
    """Run GRPO training."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, tokenizer = load_model(model_path)
    dataset = prepare_dataset(dataset_path, tokenizer, max_seq_length)

    def reward_func(completions, **kwargs):
        # TRL passes each dataset column through as a kwarg list; `answer` holds
        # the gold answers. Passing None lets combined_reward degrade gracefully
        # (format + self-correction only) if answers are ever missing.
        return combined_reward(completions, answers=kwargs.get("answer"))

    from dataclasses import fields as dataclass_fields

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
        num_generations=num_generations,
        beta=beta,
        temperature=1.0,
        top_p=0.95,
        # --- Speed: when the installed TRL supports it, generate completions
        # with a vLLM engine instead of the training model in eager mode (the
        # single biggest GRPO speedup on one GPU). Different TRL versions expose
        # these under different names, so only pass what this version accepts.
        # generation_batch_size must be divisible by num_generations, so tie it
        # to num_generations (one batched call emits the whole group).
        generation_batch_size=num_generations,
        report_to=["tensorboard"] + (["wandb"] if use_wandb else []),
        run_name="grpo-reasoner",
    )

    # Attach vLLM generation args only if GRPOConfig understands them. If the
    # installed TRL is older (no vllm_device / vllm_gpu_memory_utilization),
    # generation falls back to eager mode and we warn instead of crashing.
    _vllm_defaults = {
        "vllm_device": "cuda:0",
        "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
    }
    if vllm_kwargs:
        _vllm_defaults.update(vllm_kwargs)
    _grpo_fields = {f.name for f in dataclass_fields(GRPOConfig)}
    _vllm_supported = [k for k in _vllm_defaults if k in _grpo_fields]
    for k in _vllm_supported:
        setattr(training_args, k, _vllm_defaults[k])
    if _vllm_defaults and not _vllm_supported:
        print("WARNING: installed TRL GRPOConfig has no vLLM args "
              "(vllm_device/vllm_gpu_memory_utilization); generation will run "
              "in eager mode (slower). Upgrade TRL to enable the vLLM backend.")

    if max_steps is not None:
        training_args.max_steps = max_steps

    # Set completion length — the model needs enough room to produce
    # <thinking>...</thinking><answer>...</answer> without truncation.
    _grpo_fields = {f.name for f in dataclass_fields(GRPOConfig)}
    if "max_prompt_length" in _grpo_fields:
        training_args.max_prompt_length = max_seq_length // 2
    if "max_completion_length" in _grpo_fields:
        training_args.max_completion_length = max_seq_length
    if "max_seq_length" in _grpo_fields:
        training_args.max_seq_length = max_seq_length

    # Debug: print what TRL version supports
    print(f"GRPOConfig fields with 'max': {[f for f in _grpo_fields if 'max' in f]}")
    print(f"max_prompt_length: {getattr(training_args, 'max_prompt_length', 'N/A')}")
    print(f"max_completion_length: {getattr(training_args, 'max_completion_length', 'N/A')}")
    print(f"max_seq_length: {getattr(training_args, 'max_seq_length', 'N/A')}")

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
    parser.add_argument("--model", default="runs/sft", help="SFT checkpoint path")
    parser.add_argument("--dataset", default="data/processed/all_train.jsonl")
    parser.add_argument("--output", default="runs/grpo")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap training steps (use e.g. 50 for a quick speed/reward probe)")
    parser.add_argument("--vllm-util", type=float, default=0.3,
                        help="Fraction of GPU VRAM for the vLLM generation engine")
    parser.add_argument("--wandb", action="store_true")
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
        max_steps=args.max_steps,
        vllm_gpu_memory_utilization=args.vllm_util,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
