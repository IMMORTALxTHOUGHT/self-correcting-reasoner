# Self-Correcting Reasoning LLM Pipeline

QLoRA SFT & GRPO Alignment for Mathematical Reasoning

## Overview

End-to-end post-training pipeline that takes a lightweight open-weight model (Qwen 3.5 4B) and trains it to excel at mathematical reasoning with self-correcting capabilities.

## Key Features

- **QLoRA SFT**: Efficient fine-tuning with 4-bit quantization
- **GRPO Alignment**: Group Relative Policy Optimization without critic model
- **Self-Correction Rewards**: Custom rewards for encouraging reasoning refinement
- **Curriculum Learning**: Progressive difficulty from arithmetic to competition math
- **Evaluation**: Pass@1 accuracy, Claim-Level Reliability (CLR), format compliance

## Project Structure

```
self-correcting-reasoner/
├── data/           # Dataset preparation scripts
├── train/          # SFT and GRPO training scripts
├── reward/         # Custom reward functions
├── eval/           # Evaluation and benchmarking
├── export/         # Model export and serving
├── configs/        # Training configurations
├── scripts/        # Pipeline automation
└── docs/           # Documentation
```

## Quick Start

```bash
# 1. Setup environment
chmod +x setup.sh
./setup.sh
source venv/bin/activate

# 2. Prepare dataset
python data/gsm8k_prep.py
python data/curriculum.py

# 3. Run SFT training
python train/sft_trainer.py --config configs/sft_config.yaml

# 4. Run GRPO alignment
python train/grpo_trainer.py --config configs/grpo_config.yaml

# 5. Evaluate
python eval/benchmark.py --model runs/sft/checkpoint-1000

# 6. Export
python export/merge_lora.py --model runs/grpo/checkpoint-500
python export/quantize_gguf.py --model merged_model/
```

## Hardware Requirements

- **Minimum**: Single A10G (24GB VRAM)
- **Recommended**: Single A100 (40/80GB VRAM)

## Success Metrics

1. Pass@1 on GSM8K-test ≥ 70%
2. Format compliance ≥ 95%
3. Self-correction rate ≥ 30%
4. CLR score ≥ 0.8
5. Quantization retention ≥ 95%
