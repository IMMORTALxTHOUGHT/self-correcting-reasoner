#!/usr/bin/env python3
"""
SFT Trainer with Unsloth

Trains Qwen 3.5 4B with QLoRA on mathematical reasoning datasets.
Uses Unsloth for optimized training with reduced VRAM usage.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig


def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 16,
):
    """Load model with Unsloth optimizations and QLoRA adapters."""
    
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only
        
        print(f"Loading {model_name} with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,  # auto
        )
        
        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        
        print("Model loaded with Unsloth optimizations")
        return model, tokenizer, "unsloth"
        
    except ImportError:
        print("Unsloth not available, falling back to standard PEFT...")
        return load_model_standard(model_name, max_seq_length, lora_r, lora_alpha)


def load_model_standard(
    model_name: str,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 16,
):
    """Fallback: Load model with standard PEFT + BitsAndBytes."""
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    print(f"Loading {model_name} with standard PEFT...")
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    print("Model loaded with standard PEFT")
    return model, tokenizer, "peft"


def prepare_dataset(
    dataset_path: str,
    tokenizer,
    max_seq_length: int = 2048,
    eval_split: float = 0.1,
):
    """Load and prepare dataset for training."""
    
    print(f"Loading dataset from {dataset_path}...")
    
    # Load JSONL dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # Apply chat template
    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            # Remove BOS token to avoid duplicates
            if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
                text = text[len(tokenizer.bos_token):]
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(formatting_func, batched=True)
    
    # Split for evaluation
    if eval_split > 0:
        split = dataset.train_test_split(test_size=eval_split, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
    
    return train_dataset, eval_dataset


def train(
    config_path: Optional[str] = None,
    model_name: str = "Qwen/Qwen3-4B",
    dataset_path: str = "data/processed/gsm8k_train.jsonl",
    output_dir: str = "runs/sft",
    num_epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 8,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 16,
    eval_split: float = 0.1,
    use_wandb: bool = False,
):
    """Run SFT training."""
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    model, tokenizer, backend = load_model_and_tokenizer(
        model_name, max_seq_length, lora_r, lora_alpha
    )
    print(f"Model loaded via {backend}")
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(
        dataset_path, tokenizer, max_seq_length, eval_split
    )
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        eval_strategy="epoch" if eval_dataset else "no",
        seed=42,
        max_seq_length=max_seq_length,
        report_to=["tensorboard"] + (["wandb"] if use_wandb else []),
        run_name="sft-reasoner",
        dataset_text_field="text",
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    
    # Train on responses only (mask user inputs)
    try:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
    except ImportError:
        print("Note: train_on_responses_only not available without Unsloth")
    
    # Train
    print("\nStarting SFT training...")
    start_time = time.time()
    
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time / 60:.1f} minutes")
    
    # Print metrics
    if train_result.metrics:
        print(f"Final loss: {train_result.metrics.get('train_loss', 'N/A')}")
    
    # Save model
    print("\nSaving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nModel saved to {output_dir}")
    return train_result


def main():
    parser = argparse.ArgumentParser(description="SFT Training for Self-Correcting Reasoner")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Base model name")
    parser.add_argument("--dataset", default="data/processed/gsm8k_train.jsonl", help="Dataset path")
    parser.add_argument("--output", default="runs/sft", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Eval split ratio")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    
    args = parser.parse_args()
    
    train(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        eval_split=args.eval_split,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
