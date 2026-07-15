#!/usr/bin/env python3
"""
GGUF Quantization

Exports model to GGUF format with multiple quantization levels.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


def quantize_gguf(
    model_path: str,
    output_dir: str = "export/gguf",
    quantization_methods: list = None,
    push_to_hub: bool = False,
    hub_repo: str = None,
):
    """
    Export model to GGUF format.
    
    Args:
        model_path: Path to merged model
        output_dir: Output directory for GGUF files
        quantization_methods: List of quantization methods
        push_to_hub: Whether to push to Hub
        hub_repo: Hub repository name
    """
    if quantization_methods is None:
        quantization_methods = [
            "q4_k_m",   # 4-bit, good balance
            "q5_k_m",   # 5-bit, better quality
            "q6_k",     # 6-bit, near lossless
            "f16",      # 16-bit, no quantization
        ]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting model to GGUF format...")
    print(f"Model path: {model_path}")
    print(f"Output dir: {output_dir}")
    print(f"Quantization methods: {quantization_methods}")
    
    # Try Unsloth export first
    try:
        from unsloth import FastLanguageModel
        
        print("\nUsing Unsloth for GGUF export...")
        
        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            load_in_4bit=False,
        )
        
        # Export to each quantization method
        for method in quantization_methods:
            print(f"\nExporting with {method}...")
            
            output_file = output_path / f"model-{method}.gguf"
            
            try:
                model.save_pretrained_gguf(
                    str(output_path),
                    tokenizer,
                    quantization_method=method,
                )
                print(f"  Saved: {output_file}")
            except Exception as e:
                print(f"  Error with {method}: {e}")
        
        return
    
    except ImportError:
        print("Unsloth not available, using llama.cpp...")
    
    # Fallback: use llama.cpp
    print("\nUsing llama.cpp for GGUF export...")
    
    # Check if llama.cpp is installed
    try:
        subprocess.run(["python", "-m", "llama_cpp"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: llama.cpp not installed")
        print("Install with: pip install llama-cpp-python")
        print("Or clone and build: https://github.com/ggerganov/llama.cpp")
        sys.exit(1)
    
    # Export using llama.cpp convert script
    for method in quantization_methods:
        print(f"\nExporting with {method}...")
        
        output_file = output_path / f"model-{method}.gguf"
        
        # Build conversion command
        cmd = [
            "python", "-m", "llama_cpp",
            "convert_hf_to_gguf.py",
            model_path,
            "--outfile", str(output_file),
            "--outtype", method,
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"  Saved: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"  Error with {method}: {e}")
    
    # Push to Hub if requested
    if push_to_hub and hub_repo:
        print(f"\nPushing GGUF files to Hub: {hub_repo}...")
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            
            for method in quantization_methods:
                gguf_file = output_path / f"model-{method}.gguf"
                if gguf_file.exists():
                    api.upload_file(
                        path_or_fileobj=str(gguf_file),
                        path_in_repo=f"model-{method}.gguf",
                        repo_id=hub_repo,
                        repo_type="model",
                    )
                    print(f"  Uploaded: model-{method}.gguf")
        
        except ImportError:
            print("ERROR: huggingface_hub not installed")
            print("Install with: pip install huggingface_hub")
    
    # Save metadata
    metadata = {
        "model_path": model_path,
        "output_dir": output_dir,
        "quantization_methods": quantization_methods,
        "pushed_to_hub": push_to_hub,
        "hub_repo": hub_repo,
    }
    
    metadata_file = output_path / "quantization_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="GGUF Quantization")
    parser.add_argument("--model", required=True, help="Path to merged model")
    parser.add_argument("--output", default="export/gguf", help="Output directory")
    parser.add_argument("--methods", nargs="+", default=["q4_k_m", "q5_k_m", "q6_k"],
                        help="Quantization methods")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to Hub")
    parser.add_argument("--hub-repo", help="Hub repository name")
    
    args = parser.parse_args()
    
    quantize_gguf(
        model_path=args.model,
        output_dir=args.output,
        quantization_methods=args.methods,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
    )


if __name__ == "__main__":
    main()
