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
# CAT tick 2026-09-27_18:51:25 tick=1790535085
# CAT tick 2026-09-28_12:30:28 tick=1790598628
# CAT tick 2026-09-29_09:17:01 tick=1790673421
# CAT tick 2026-09-29_09:30:35 tick=1790674235
# CAT tick 2026-09-29_12:30:34 tick=1790685034
# CAT tick 2026-09-30_06:30:19 tick=1790749819
# CAT tick 2026-09-30_10:40:59 tick=1790764859
# CAT tick 2026-09-30_14:45:48 tick=1790779548
# CAT tick 2026-10-02_15:41:38 tick=1790955698
# CAT tick 2026-10-03_12:14:29 tick=1791029669
# CAT tick 2026-10-03_12:14:31 tick=1791029671
# CAT tick 2026-10-03_12:14:31 tick=1791029671
# CAT tick 2026-10-03_12:30:21 tick=1791030621
# CAT tick 2026-10-03_12:30:26 tick=1791030626

# CAT tick 2026-10-04_09:24:44

# CAT tick 2026-10-06_11:08:48

# CAT tick 2026-10-07_14:21:21

# CAT tick 2026-10-07_14:21:33

# CAT tick 2026-10-07_14:21:39

# CAT tick 2026-10-08_16:09:10

# CAT tick 2026-10-09_13:33:45

# CAT tick 2026-10-09_13:34:35
