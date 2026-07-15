#!/bin/bash
# End-to-end pipeline for Self-Correcting Reasoner

set -e

echo "=========================================="
echo "Self-Correcting Reasoner Pipeline"
echo "=========================================="

# Configuration
MODEL="Qwen/Qwen3-4B"
OUTPUT_DIR="runs"
DATA_DIR="data/processed"

# Step 1: Setup environment
echo -e "\n[1/7] Setting up environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt

# Step 2: Prepare datasets
echo -e "\n[2/7] Preparing datasets..."
python data/gsm8k_prep.py
python data/math_prep.py
python data/curriculum.py

# Step 3: Run SFT training
echo -e "\n[3/7] Running SFT training..."
python train/sft_trainer.py \
    --model "$MODEL" \
    --dataset "$DATA_DIR/gsm8k_train.jsonl" \
    --output "$OUTPUT_DIR/sft" \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 8

# Step 4: Run GRPO alignment
echo -e "\n[4/7] Running GRPO alignment..."
python train/grpo_trainer.py \
    --model "$OUTPUT_DIR/sft" \
    --dataset "$DATA_DIR/gsm8k_train.jsonl" \
    --output "$OUTPUT_DIR/grpo" \
    --num-generations 8 \
    --beta 0.1

# Step 5: Evaluate
echo -e "\n[5/7] Evaluating model..."
python eval/benchmark.py \
    --model "$OUTPUT_DIR/grpo" \
    --dataset "$DATA_DIR/gsm8k_test.jsonl" \
    --generations 64

# Step 6: Export
echo -e "\n[6/7] Exporting model..."
python export/merge_lora.py \
    --model "$OUTPUT_DIR/grpo" \
    --output "export/merged"

python export/quantize_gguf.py \
    --model "export/merged" \
    --output "export/gguf" \
    --methods q4_k_m q5_k_m q6_k

# Step 7: Serve
echo -e "\n[7/7] Setting up serving..."
python export/serve_ollama.py \
    --model-path "export/gguf/model-q5_k_m.gguf" \
    --model-name "self-correcting-reasoner" \
    --action create

echo -e "\n=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Test model: ollama run self-correcting-reasoner"
echo "  2. Or serve with vLLM: python export/serve_vllm.py --model export/merged"
echo ""
