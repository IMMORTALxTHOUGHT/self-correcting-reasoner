#!/usr/bin/env python3
"""
SFT Trainer using PEFT + BitsAndBytes QLoRA.

Trains Qwen 3.5 4B on mathematical reasoning datasets.
"""

import argparse
import time
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def load_model(model_name: str = "Qwen/Qwen3.5-2B", lora_r: int = 16, lora_alpha: int = 16):
    """Load model with 4-bit quantization and LoRA adapters."""

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def prepare_dataset(dataset_path: str, tokenizer, max_seq_length: int = 2048, eval_split: float = 0.1):
    """Load and format dataset for SFT."""

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def format(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]
        return {"text": text}

    dataset = dataset.map(format)

    if eval_split > 0:
        split = dataset.train_test_split(test_size=eval_split, seed=42)
        return split["train"], split["test"]
    return dataset, None


def train(
    model_name: str = "Qwen/Qwen3.5-2B",
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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, tokenizer = load_model(model_name, lora_r, lora_alpha)
    train_dataset, eval_dataset = prepare_dataset(dataset_path, tokenizer, max_seq_length, eval_split)

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
        report_to=["tensorboard"] + (["wandb"] if use_wandb else []),
        run_name="sft-reasoner",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
    )

    print("Starting SFT training...")
    start = time.time()
    result = trainer.train()
    print(f"Done in {(time.time() - start) / 60:.1f} min")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--dataset", default="data/processed/all_train.jsonl")
    parser.add_argument("--output", default="runs/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--wandb", action="store_true")
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
