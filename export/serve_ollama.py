#!/usr/bin/env python3
"""
Ollama Integration

Integrates the fine-tuned model with Ollama for local serving.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def create_modelfile(model_path, output_path="Modelfile", model_name="self-correcting-reasoner"):
    """Create Ollama Modelfile."""
    
    system_prompt = (
        "You are a mathematical reasoning assistant. "
        "Always think step by step in <thinking> tags and "
        "provide your final answer in <answer> tags."
    )
    
    # Write Modelfile content
    content = f"""FROM {model_path}

PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER top_k -1

SYSTEM {system_prompt}
"""
    
    with open(output_path, "w") as f:
        f.write(content)
    
    print(f"Modelfile created at {output_path}")
    return output_path


def pull_model(model_name):
    """Pull model from Ollama registry."""
    print(f"Pulling {model_name}...")
    subprocess.run(["ollama", "pull", model_name], check=True)


def create_ollama_model(model_name, modelfile_path="Modelfile"):
    """Create Ollama model from Modelfile."""
    print(f"Creating Ollama model: {model_name}")
    subprocess.run(["ollama", "create", model_name, "-f", modelfile_path], check=True)
    print(f"Model {model_name} created successfully")


def serve_model(model_name, host="localhost", port=11434):
    """Start Ollama server."""
    print(f"Starting Ollama server...")
    print(f"Model: {model_name}")
    print(f"Host: {host}:{port}")
    
    subprocess.run(["ollama", "serve"], check=True)


def test_model(model_name, prompt="What is 2 + 3 * 4?"):
    """Test the model with a sample prompt."""
    print(f"Testing model: {model_name}")
    print(f"Prompt: {prompt}")
    
    result = subprocess.run(
        ["ollama", "run", model_name, prompt],
        capture_output=True,
        text=True,
        check=True,
    )
    
    print(f"\nResponse:\n{result.stdout}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Ollama Integration")
    parser.add_argument("--model-path", help="Path to GGUF model")
    parser.add_argument("--model-name", default="self-correcting-reasoner", help="Ollama model name")
    parser.add_argument("--action", choices=["create", "serve", "test"], default="create")
    parser.add_argument("--prompt", default="What is 2 + 3 * 4?", help="Test prompt")
    
    args = parser.parse_args()
    
    if args.action == "create":
        if not args.model_path:
            print("ERROR: --model-path required for create action")
            sys.exit(1)
        
        modelfile = create_modelfile(args.model_path, model_name=args.model_name)
        create_ollama_model(args.model_name, modelfile)
        
    elif args.action == "serve":
        serve_model(args.model_name)
        
    elif args.action == "test":
        test_model(args.model_name, args.prompt)


if __name__ == "__main__":
    main()
