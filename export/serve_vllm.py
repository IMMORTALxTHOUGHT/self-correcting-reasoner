#!/usr/bin/env python3
"""
vLLM Serving Script

Serves the fine-tuned model using vLLM for high-throughput inference.
"""

import argparse
import json
import sys
from pathlib import Path


def serve_vllm(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
    trust_remote_code: bool = True,
):
    """
    Serve model using vLLM.
    
    Args:
        model_path: Path to model
        host: Host to bind to
        port: Port to bind to
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_model_len: Maximum model length
        gpu_memory_utilization: GPU memory utilization
        trust_remote_code: Trust remote code
    """
    try:
        from vllm import LLM, SamplingParams
        from vllm.entrypoints.openai.api_server import app
        import uvicorn
        
    except ImportError:
        print("ERROR: vLLM not installed")
        print("Install with: pip install vllm")
        sys.exit(1)
    
    print(f"Starting vLLM server...")
    print(f"Model: {model_path}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Tensor parallel: {tensor_parallel_size}")
    
    # Set environment variables
    import os
    os.environ["VLLM_HOST_IP"] = host
    os.environ["VLLM_PORT"] = str(port)
    
    # Start server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


def test_inference(model_path: str, prompt: str = "What is 2 + 3 * 4?"):
    """Test inference with the model."""
    try:
        from vllm import LLM, SamplingParams
        
        print(f"Loading model from {model_path}...")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            max_model_len=4096,
        )
        
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.95,
            max_tokens=1024,
        )
        
        print(f"\nGenerating response for: {prompt}")
        outputs = llm.generate([prompt], sampling_params)
        
        response = outputs[0].outputs[0].text
        print(f"\nResponse:\n{response}")
        
        return response
        
    except ImportError:
        print("ERROR: vLLM not installed")
        return None


def main():
    parser = argparse.ArgumentParser(description="vLLM Serving")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model length")
    parser.add_argument("--gpu-memory", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--test", action="store_true", help="Test inference only")
    
    args = parser.parse_args()
    
    if args.test:
        test_inference(args.model)
    else:
        serve_vllm(
            model_path=args.model,
            host=args.host,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory,
        )


if __name__ == "__main__":
    main()
