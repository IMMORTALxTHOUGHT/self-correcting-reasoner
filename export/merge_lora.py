#!/usr/bin/env python3
"""
Merge LoRA Adapters

Merges LoRA adapters back into the base model for deployment.
Supports chained adapters (e.g. runs/grpo -> runs/sft -> Qwen/Qwen3.5-2B).
"""

import argparse
import json
from pathlib import Path

import torch


def merge_lora(
    model_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_repo: str | None = None,
) -> None:
    """
    Merge LoRA adapters into base model.

    Args:
        model_path: Path to model/LoRA adapter. If the adapter's
            base_model_name_or_path is itself a LoRA dir (e.g. GRPO on top
            of SFT), the chain is resolved automatically to the real HF base.
        output_path: Directory to save the merged full model.
        push_to_hub: If True, push the merged model to the HuggingFace Hub.
        hub_repo: Hub repo id (e.g. USERNAME/my-model), required if push_to_hub.
    """
    print(f"Loading model from {model_path}...")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Resolve the *real* base model by climbing adapter_config.json chains.
    adapter_chain: list[str] = []
    current = Path(model_path)
    while (current / "adapter_config.json").exists():
        with open(current / "adapter_config.json") as f:
            base = json.load(f).get("base_model_name_or_path", str(current))
        adapter_chain.append(str(current))
        current = Path(base)
    base_model_name = str(current)

    print(f"  Base model: {base_model_name}")
    print(f"  Adapter chain: {' -> '.join(adapter_chain)}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Stack each adapter in chain order, then merge them all into the base.
    for adapter_dir in adapter_chain:
        base_model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = base_model.merge_and_unload()

    print("Model merged via PEFT")

    # Save merged model
    print(f"\nSaving merged model to {output_path}...")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to {output_path}")

    # Push to Hub if requested
    if push_to_hub and hub_repo:
        print(f"\nPushing to Hub: {hub_repo}...")
        try:
            model.push_to_hub(hub_repo, tokenizer=tokenizer)
            print(f"  Pushed: {hub_repo}")
        except Exception as e:
            print(f"  ERROR pushing to hub: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--model", required=True,
                        help="Path to model/LoRA adapter (supports chains like runs/grpo)")
    parser.add_argument("--output", required=True,
                        help="Output directory for merged model")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push merged model to HuggingFace Hub")
    parser.add_argument("--hub-repo",
                        help="Hub repo id (e.g. USERNAME/my-model)")
    args = parser.parse_args()

    merge_lora(
        model_path=args.model,
        output_path=args.output,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
    )


if __name__ == "__main__":
    main()

