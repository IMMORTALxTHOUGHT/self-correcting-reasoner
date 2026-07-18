#!/usr/bin/env python3
"""
Merge LoRA Adapters

Merges LoRA adapters back into the base model for deployment.
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def merge_lora(
    model_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_repo: str = None,
):
    """
    Merge LoRA adapters into base model.

    Args:
        model_path: Path to model with LoRA adapters
        output_path: Path to save merged model
        push_to_hub: Whether to push to Hugging Face Hub
        hub_repo: Hub repository name
    """
    print(f"Loading model from {model_path}...")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Resolve the *real* base model by climbing adapter_config.json chains.
    # A GRPO adapter's base may itself be an SFT adapter dir (also a LoRA),
    # whose base is the actual HF model. Collect the chain of adapter dirs in
    # order (outermost first), ending at the true base model name/path.
    adapter_chain = []
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

    print("Model merged via peft")

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
        model.push_to_hub(hub_repo, tokenizer=tokenizer)
        print(f"Model pushed to https://huggingface.co/{hub_repo}")

    # Save metadata
    metadata = {
        "model_path": model_path,
        "output_path": output_path,
        "base_model": base_model_name,
        "pushed_to_hub": push_to_hub,
        "hub_repo": hub_repo,
    }

    metadata_file = output_dir / "merge_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to {metadata_file}")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA Adapters")
    parser.add_argument("--model", required=True, help="Path to model with LoRA")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to Hub")
    parser.add_argument("--hub-repo", help="Hub repository name")

    args = parser.parse_args()

    if args.push_to_hub and not args.hub_repo:
        print("ERROR: --hub-repo required when using --push-to-hub")
        sys.exit(1)

    merge_lora(
        model_path=args.model,
        output_path=args.output,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
    )


if __name__ == "__main__":
    main()
