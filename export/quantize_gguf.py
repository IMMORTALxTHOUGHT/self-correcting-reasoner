#!/usr/bin/env python3
"""
GGUF Quantization

Exports model to GGUF format with multiple quantization levels.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Quant types convert_hf_to_gguf.py can emit directly via --outtype.
# Everything else (K-quants like Q4_K_M) must go through the llama-quantize
# binary applied to a high-precision base GGUF.
DIRECT_OUTTYPES = {"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"}


def _find_llama_cpp_dir() -> Path:
    """Locate the llama.cpp checkout (env override or sibling of project root)."""
    env = os.environ.get("LLAMA_CPP_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "llama.cpp"


def _find_convert_script(llama_dir: Path):
    candidate = llama_dir / "convert_hf_to_gguf.py"
    return candidate if candidate.exists() else None


def _find_quantize_bin(llama_dir: Path):
    """Find the llama-quantize binary via env, PATH, or a llama.cpp build dir."""
    env = os.environ.get("LLAMA_QUANTIZE")
    if env and Path(env).exists():
        return env
    found = shutil.which("llama-quantize") or shutil.which("quantize")
    if found:
        return found
    for rel in ("build/bin/llama-quantize", "build/bin/quantize", "llama-quantize", "quantize"):
        p = llama_dir / rel
        if p.exists():
            return str(p)
    return None


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
        quantization_methods = ["q4_k_m", "q8_0", "bf16"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    methods = [m.lower() for m in quantization_methods]

    print("Exporting model to GGUF format...")
    print(f"Model path: {model_path}")
    print(f"Output dir: {output_dir}")
    print(f"Quantization methods: {methods}")

    llama_dir = _find_llama_cpp_dir()
    convert_script = _find_convert_script(llama_dir)
    if convert_script is None:
        print("\nERROR: convert_hf_to_gguf.py not found.")
        print(f"  Looked under: {llama_dir}")
        print("  Clone llama.cpp there or set LLAMA_CPP_DIR, e.g.:")
        print("    git clone https://github.com/ggml-org/llama.cpp")
        sys.exit(1)

    # K-quants (anything not directly emittable by the converter) require the
    # llama-quantize binary applied to a high-precision base GGUF.
    needs_quantize = any(m not in DIRECT_OUTTYPES for m in methods)
    quantize_bin = _find_quantize_bin(llama_dir) if needs_quantize else None
    if needs_quantize and quantize_bin is None:
        print("\nERROR: llama-quantize binary not found (required for K-quants like q4_k_m).")
        print(f"  Searched PATH and under: {llama_dir}")
        print("  Build llama.cpp (cmake -B build && cmake --build build -j) or set LLAMA_QUANTIZE.")
        sys.exit(1)

    # Build the f16 base GGUF once, reused for every K-quant target.
    base_gguf = None
    if needs_quantize:
        base_gguf = output_path / "model-f16.gguf"
        if base_gguf.exists():
            print(f"\nReusing existing base GGUF: {base_gguf}")
        else:
            print("\nConverting HF model -> base f16 GGUF...")
            subprocess.run(
                [sys.executable, str(convert_script), model_path,
                 "--outfile", str(base_gguf), "--outtype", "f16"],
                check=True,
            )

    for method in methods:
        output_file = output_path / f"model-{method}.gguf"
        print(f"\nExporting with {method} -> {output_file}")

        if method in DIRECT_OUTTYPES:
            # Emitted directly from the HF checkpoint by the converter.
            subprocess.run(
                [sys.executable, str(convert_script), model_path,
                 "--outfile", str(output_file), "--outtype", method],
                check=True,
            )
        else:
            # Two-step: base f16 GGUF -> llama-quantize -> target K-quant.
            subprocess.run(
                [quantize_bin, str(base_gguf), str(output_file), method.upper()],
                check=True,
            )

        print(f"  Saved: {output_file}")
    
    # Push to Hub if requested
    if push_to_hub and hub_repo:
        print(f"\nPushing GGUF files to Hub: {hub_repo}...")
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            
            for method in methods:
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
        "quantization_methods": methods,
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
    parser.add_argument("--methods", nargs="+", default=["q4_k_m", "q8_0", "bf16"],
                        help="Quantization methods (e.g. q4_k_m q5_k_m q6_k q8_0 f16 bf16)")
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
