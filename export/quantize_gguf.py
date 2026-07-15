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
    
    # Export using llama.cpp convert script (or ctransformers as fallback)
    
    # Export using llama.cpp convert script (or ctransformers as fallback)
    for method in quantization_methods:
        print(f"\nExporting with {method}...")

        output_file = output_path / f"model-{method}.gguf"

        # Try the llama.cpp convert_hf_to_gguf.py script first
        try:
            cmd = [
                "python3", str(Path(__file__).resolve().parent.parent / "llama.cpp" / "convert_hf_to_gguf.py"),
                model_path,
                "--outfile", str(output_file),
                "--outtype", method,
            ]
            subprocess.run(cmd, check=True)
            print(f"  Saved: {output_file}")
            continue
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Fallback: try ctransformers
        try:
            from ctransformers import AutoModelForCausalLM as CTAutoModel
            print(f"  ctransformers available — loading model for quantization...")
            ct_model = CTAutoModel.from_pretrained(model_path, model_type="llama")
            ct_model.save_pretrained(str(output_file), quantization_method=method)
            print(f"  Saved: {output_file}")
            continue
        except Exception:
            pass

        # Last resort: huggingface-gguf conversion
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import struct

            print(f"  Using transformers fallback for {method}...")
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            # Save as safetensors first, then we'd need gguf conversion
            tmp_dir = output_path / f"_tmp_{method}"
            model.save_pretrained(str(tmp_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(tmp_dir))
            print(f"  Saved safetensors to {tmp_dir} — manual GGUF conversion needed for {method}")
            print(f"  Run: python convert_hf_to_gguf.py {tmp_dir} --outfile {output_file} --outtype {method}")
        except Exception as e:
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
