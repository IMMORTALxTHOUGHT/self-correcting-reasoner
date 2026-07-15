# Methodology

## Overview

This project implements a self-correcting reasoning LLM pipeline using QLoRA SFT and GRPO alignment for mathematical reasoning.

## Training Pipeline

### Phase 1: Dataset Preparation

1. **GSM8K Processing**: Extract question-answer pairs and format with `<thinking>` and `<answer>` tags
2. **MATH Processing**: Handle LaTeX notation and competition-level problems
3. **Curriculum Stratification**: Split data by difficulty (easy/medium/hard)

### Phase 2: SFT Training

- **Model**: Qwen 3.5 4B (or fallback to Qwen 2.5 7B)
- **Method**: QLoRA with 4-bit quantization
- **LoRA Config**: r=16, alpha=16, targeting attention and MLP layers
- **Training**: 3 epochs, batch size 2, gradient accumulation 8

### Phase 3: GRPO Alignment

- **Reward Functions**:
  - Format reward (0.3): Proper `<thinking>` and `<answer>` tags
  - Self-correction reward (0.3): Backtracking patterns in reasoning
  - Accuracy reward (0.4): Correct final answer
- **Group Size**: 8 generations per prompt
- **KL Penalty**: beta=0.1

### Phase 4: Evaluation

- **Pass@1**: Mean accuracy over 64 generations
- **CLR**: Claim-Level Reliability with K=32 traces, M=5 claims
- **Format Compliance**: Percentage of properly formatted outputs

## Key Design Decisions

1. **Mathematical Reasoning**: Chosen for verifiable rewards (exact answer matching)
2. **Self-Correction**: Structured patterns to detect genuine backtracking
3. **Curriculum Learning**: Progressive difficulty for better learning
4. **Single GPU Focus**: Optimized for A10G (24GB) without FSDP2 overhead

## References

- QLoRA: Dettmers et al. (2023)
- GRPO: Shao et al. (2024)
- GSM8K: Cobbe et al. (2021)
- MATH: Hendrycks et al. (2021)
