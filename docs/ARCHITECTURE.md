# ARCHITECTURE.md - Complete Technical Guide

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Concepts](#2-core-concepts)
3. [System Architecture](#3-system-architecture)
4. [Data Pipeline](#4-data-pipeline)
5. [QLoRA SFT Training](#5-qlora-sft-training)
6. [GRPO Alignment](#6-grpo-alignment)
7. [Reward Functions](#7-reward-functions)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Model Export & Serving](#9-model-export--serving)
10. [Code Walkthrough](#10-code-walkthrough)
11. [Configuration Reference](#11-configuration-reference)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Project Overview

### What This Project Does

This project takes a base language model (Qwen 3.5 4B) and post-trains it to excel at mathematical reasoning with self-correcting capabilities. The pipeline has two main training phases:

1. **SFT (Supervised Fine-Tuning)**: Teaches the model formatting and domain knowledge
2. **GRPO (Group Relative Policy Optimization)**: Aligns the model to produce logical, self-correcting reasoning paths

### Why Mathematical Reasoning?

Mathematical reasoning is ideal for this project because:
- **Verifiable rewards**: Exact answer matching provides clear training signal
- **Structured output**: Problems have clear steps and final answers
- **Self-correction natural**: Mistakes in math are easy to detect and correct

### End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│  GSM8K/MATH → Format with tags → Curriculum split → JSONL       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SFT TRAINING (Phase 1)                      │
│  Qwen 3.5 4B + QLoRA → Train on formatted data → Checkpoint     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GRPO ALIGNMENT (Phase 2)                     │
│  SFT Checkpoint → Sample groups → Reward scoring → Update model  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION & EXPORT                          │
│  Pass@1, CLR metrics → Merge LoRA → GGUF quantization → Serve   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Concepts

### 2.1 QLoRA (Quantized Low-Rank Adaptation)

**What it is**: A method to fine-tune large models on consumer GPUs by combining 4-bit quantization with LoRA adapters.

**How it works**:
1. **4-bit Quantization**: The base model weights are quantized from 16-bit to 4-bit, reducing memory by ~75%
2. **LoRA Adapters**: Small trainable matrices (rank r) are added to attention layers
3. **Training**: Only LoRA adapters are trained; base weights stay frozen

**Key Parameters**:
```python
lora_config = {
    "r": 16,              # Rank of adaptation matrices (higher = more capacity)
    "lora_alpha": 16,     # Scaling factor (typically = r)
    "target_modules": [   # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ],
    "lora_dropout": 0.0,  # Dropout for regularization
}
```

**Why QLoRA**:
- A 4B parameter model in FP16 needs ~8GB VRAM
- With 4-bit quantization: ~2GB VRAM
- LoRA adapters add only ~10MB
- Total: ~2.5GB VRAM (fits easily on A10G)

### 2.2 SFT (Supervised Fine-Tuning)

**What it is**: Training the model on input-output pairs to learn a specific task/format.

**In this project**:
- Input: Mathematical problem
- Output: `<thinking>` step-by-step reasoning + `<answer>` final answer

**Example**:
```json
{
  "messages": [
    {"role": "user", "content": "What is 2 + 3 * 4?"},
    {"role": "assistant", "content": "<thinking>\nFirst, multiplication: 3 * 4 = 12\nThen addition: 2 + 12 = 14\n</thinking>\n<answer>\n14\n</answer>"}
  ]
}
```

### 2.3 GRPO (Group Relative Policy Optimization)

**What it is**: A reinforcement learning algorithm that trains without a separate critic model.

**How it works**:
1. **Sample**: For each prompt, generate K completions (the "group")
2. **Reward**: Score each completion using reward functions
3. **Baseline**: Calculate group-relative baseline (mean reward)
4. **Update**: Increase probability of above-average completions, decrease below-average

**Key advantage**: No critic model needed (saves memory and compute)

**Formula**:
```
L_GRPO = -E[r(x,y) * (log π(y|x) - log π_old(y|x))]
        + β * KL(π || π_ref)
```

Where:
- `r(x,y)` = reward for prompt x, completion y
- `π` = current policy
- `π_ref` = reference policy (for KL penalty)
- `β` = KL penalty coefficient

### 2.4 Curriculum Learning

**What it is**: Training on progressively harder examples.

**In this project**:
1. **Stage 1 (Easy)**: Simple arithmetic, 1-2 step problems
2. **Stage 2 (Medium)**: Multi-step problems, word problems
3. **Stage 3 (Hard)**: Competition-level math, proofs

**Why**: Models learn foundational patterns first, then build complexity.

### 2.5 Self-Correction

**What it is**: The model's ability to recognize and fix its own reasoning mistakes.

**Example**:
```
<thinking>
Let me solve: What is 15 * 12?
15 * 12 = 180

Wait, let me recalculate to verify.
15 * 10 = 150
15 * 2 = 30
150 + 30 = 180

Yes, 180 is correct.
</thinking>
<answer>
180
</answer>
```

**Why it matters**: Genuine self-correction indicates deeper understanding.

---

## 3. System Architecture

### 3.1 Directory Structure

```
self-correcting-reasoner/
│
├── data/                          # Data preparation
│   ├── gsm8k_prep.py             # GSM8K dataset processing
│   ├── math_prep.py              # MATH dataset processing
│   ├── synthetic_traces.py       # Generate reasoning traces
│   └── curriculum.py             # Difficulty stratification
│
├── train/                         # Training scripts
│   ├── sft_config.py             # SFT hyperparameters
│   ├── sft_trainer.py            # SFT training loop
│   ├── grpo_config.py            # GRPO hyperparameters
│   └── grpo_trainer.py           # GRPO training loop
│
├── reward/                        # Reward functions
│   ├── format_reward.py          # Format compliance
│   ├── self_correction_reward.py # Backtracking detection
│   ├── accuracy_reward.py        # Answer verification
│   └── combined_reward.py        # Weighted combination
│
├── eval/                          # Evaluation
│   ├── benchmark.py              # Pass@1 accuracy
│   └── clr_eval.py              # Claim-Level Reliability
│
├── export/                        # Model deployment
│   ├── merge_lora.py             # Merge LoRA weights
│   ├── quantize_gguf.py          # GGUF quantization
│   ├── serve_vllm.py             # vLLM serving
│   └── serve_ollama.py           # Ollama integration
│
├── configs/                       # Configuration
│   ├── sft_config.yaml           # SFT YAML config
│   └── grpo_config.yaml          # GRPO YAML config
│
├── scripts/                       # Automation
│   └── run_pipeline.sh           # End-to-end pipeline
│
└── docs/                          # Documentation
    ├── ARCHITECTURE.md           # This file
    ├── methodology.md            # Methodology summary
    └── results.md                # Experimental results
```

### 3.2 Data Flow

```
Raw Datasets (GSM8K, MATH)
         │
         ▼
┌─────────────────────────┐
│   Data Preparation      │
│   - Format with tags    │
│   - Clean/Lint          │
│   - Add metadata        │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Curriculum Split      │
│   - Easy (1-2 steps)    │
│   - Medium (3-5 steps)  │
│   - Hard (6+ steps)     │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   JSONL Files           │
│   - gsm8k_train.jsonl   │
│   - gsm8k_test.jsonl    │
│   - curriculum/*.jsonl  │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Training              │
│   - SFT (3 epochs)      │
│   - GRPO (1 epoch)      │
└─────────────────────────┘
```

### 3.3 Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Base Model (Qwen 3.5 4B)                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Transformer Layers                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │  Attention   │  │    MLP      │  │  LayerNorm  │  │   │
│  │  │  ┌───────┐  │  │  ┌───────┐  │  │             │  │   │
│  │  │  │Q Proj │  │  │  │Gate   │  │  │             │  │   │
│  │  │  │K Proj │  │  │  │Up     │  │  │             │  │   │
│  │  │  │V Proj │  │  │  │Down   │  │  │             │  │   │
│  │  │  │O Proj │  │  │  └───────┘  │  │             │  │   │
│  │  │  └───────┘  │  │             │  │             │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    QLoRA Adapters                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LoRA matrices (r=16) added to:                     │   │
│  │  - q_proj, k_proj, v_proj, o_proj (Attention)      │   │
│  │  - gate_proj, up_proj, down_proj (MLP)              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Data Pipeline

### 4.1 GSM8K Dataset

**Source**: OpenAI's Grade School Math 8K

**Format**:
```json
{
  "question": "Janet's ducks lay 16 eggs per day...",
  "answer": "She bakes 4 dozen muffins...\n#### 60"
}
```

**Processing** (`data/gsm8k_prep.py`):
1. Load dataset from HuggingFace
2. Extract numerical answer after `####`
3. Format with `<thinking>` and `<answer>` tags
4. Save as JSONL

**Output format**:
```json
{
  "messages": [
    {"role": "user", "content": "Janet's ducks lay 16 eggs..."},
    {"role": "assistant", "content": "<thinking>\nJanet has 16 ducks...\n</thinking>\n<answer>\n60\n</answer>"}
  ],
  "metadata": {"dataset": "gsm8k", "answer": "60"}
}
```

### 4.2 Curriculum Stratification

**File**: `data/curriculum.py`

**Heuristics for difficulty**:
```python
def estimate_difficulty(record):
    score = 0
    
    # Question length
    if len(question) > 200: score += 2
    elif len(question) > 100: score += 1
    
    # Reasoning steps
    if step_count > 10: score += 2
    elif step_count > 5: score += 1
    
    # Complex operations (integral, matrix, etc.)
    if has_complex_ops: score += 1
    
    # Self-correction (indicates complexity)
    if has_backtracking: score += 1
    
    if score <= 2: return "easy"
    elif score <= 4: return "medium"
    else: return "hard"
```

### 4.3 Synthetic Trace Generation

**File**: `data/synthetic_traces.py`

**Purpose**: Generate high-quality reasoning traces using a teacher model.

**Process**:
1. Send problem to teacher model (GPT-4o/Claude)
2. Request step-by-step solution with self-correction
3. Parse response into `<thinking>` and `<answer>` format
4. Add quality filters

---

## 5. QLoRA SFT Training

### 5.1 Model Loading

**File**: `train/sft_trainer.py`

```python
# Load with Unsloth (optimized)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-4B",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    use_gradient_checkpointing="unsloth",
)
```

### 5.2 Training Configuration

```python
training_args = SFTConfig(
    output_dir="runs/sft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,    # Effective batch = 16
    learning_rate=2e-4,
    num_train_epochs=3,
    warmup_ratio=0.03,
    optim="adamw_8bit",
    bf16=True,
    max_seq_length=2048,
)
```

### 5.3 Response-Only Training

**Important**: Only train on assistant responses, not user prompts.

```python
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
```

**Why**: We don't want the model to learn to generate user prompts.

### 5.4 VRAM Management

**A10G (24GB) allocation**:
- Base model (4-bit): ~2GB
- LoRA adapters: ~100MB
- Optimizer states: ~500MB
- Activations (with grad checkpointing): ~10GB
- Batch data: ~2GB
- **Total**: ~13GB (fits comfortably)

**Optimizations**:
- 4-bit quantization (NF4)
- Gradient checkpointing
- AdamW 8-bit optimizer
- Small batch size with gradient accumulation

---

## 6. GRPO Alignment

### 6.1 How GRPO Works

**Step 1: Sample Group**
```python
# For each prompt, generate K=8 completions
completions = model.generate(prompt, n=8)
```

**Step 2: Score Rewards**
```python
# Score each completion
rewards = [reward_func(comp) for comp in completions]
# rewards = [0.8, 0.6, 0.9, 0.5, 0.7, 0.85, 0.65, 0.75]
```

**Step 3: Calculate Baseline**
```python
# Group-relative baseline (mean reward)
baseline = mean(rewards)
# baseline = 0.73
```

**Step 4: Policy Update**
```python
# For each completion:
# - If reward > baseline: increase probability
# - If reward < baseline: decrease probability

for comp, reward in zip(completions, rewards):
    advantage = reward - baseline
    # Update policy proportionally to advantage
```

### 6.2 GRPO vs PPO

| Aspect | PPO | GRPO |
|--------|-----|------|
| Critic Model | Required (same size as policy) | Not needed |
| Memory | ~2x model size | ~1x model size |
| Stability | More stable | Can be unstable |
| Compute | Higher | Lower |

**Why GRPO for this project**: Single A10G can't fit both policy + critic models.

### 6.3 KL Penalty

**Purpose**: Prevent model from diverging too far from reference policy.

```python
# KL penalty term
kl_penalty = beta * KL(π_current || π_reference)

# Beta = 0.1 (conservative)
# Higher beta = more conservative updates
# Lower beta = more aggressive updates
```

---

## 7. Reward Functions

### 7.1 Format Reward

**File**: `reward/format_reward.py`

**Purpose**: Ensure model outputs proper `<thinking>` and `<answer>` tags.

```python
def format_reward(completions):
    rewards = []
    for comp in completions:
        content = comp[0]["content"]
        score = 0.0
        
        # Check for thinking tags
        if "<thinking>" in content and "</thinking>" in content:
            score += 0.4
        
        # Check for answer tags
        if "<answer>" in content and "</answer>" in content:
            score += 0.4
        
        # Check for structure
        if content.count("\n") >= 2:
            score += 0.2
        
        rewards.append(score)
    return rewards
```

### 7.2 Self-Correction Reward

**File**: `reward/self_correction_reward.py`

**Purpose**: Reward genuine self-correction in reasoning.

```python
BACKTRACKING_PATTERNS = [
    r"wait,? let me",
    r"actually,? i (made|was) (a )?mistake",
    r"let me (rethink|recalculate|reconsider)",
    r"correction:",
    r"scratch that",
]

def self_correction_reward(completions):
    rewards = []
    for comp in completions:
        content = comp[0]["content"]
        
        # Extract thinking block
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
        
        if thinking_match:
            thinking = thinking_match.group(1)
            
            # Check for backtracking
            has_backtracking = any(
                re.search(p, thinking, re.IGNORECASE)
                for p in BACKTRACKING_PATTERNS
            )
            score = 0.5 if has_backtracking else 0.0
            
            # Bonus for structured correction
            if "→" in thinking or "therefore" in thinking.lower():
                score += 0.3
            
            # Bonus for multiple steps
            if thinking.count("\n") >= 3:
                score += 0.2
            
            rewards.append(min(score, 1.0))
        else:
            rewards.append(0.0)
    
    return rewards
```

### 7.3 Accuracy Reward

**File**: `reward/accuracy_reward.py`

**Purpose**: Reward correct final answers.

```python
def accuracy_reward(completions, answers):
    rewards = []
    for comp, gold in zip(completions, answers):
        content = comp[0]["content"]
        
        # Extract answer
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        
        if answer_match:
            pred = answer_match.group(1).strip()
            
            # Use math_verify for exact matching
            try:
                from math_verify import parse, verify
                ok = verify(parse(str(gold)), parse(pred))
            except:
                ok = pred.strip() == str(gold).strip()
            
            rewards.append(1.0 if ok else 0.0)
        else:
            rewards.append(0.0)
    
    return rewards
```

### 7.4 Combined Reward

**File**: `reward/combined_reward.py`

**Purpose**: Weighted combination of all rewards.

```python
def combined_reward(completions, answers, weights=None):
    if weights is None:
        weights = {
            "format": 0.3,
            "self_correction": 0.3,
            "accuracy": 0.4,
        }
    
    format_rewards = format_reward(completions)
    correction_rewards = self_correction_reward(completions)
    accuracy_rewards = accuracy_reward(completions, answers)
    
    combined = []
    for f, c, a in zip(format_rewards, correction_rewards, accuracy_rewards):
        score = (
            weights["format"] * f +
            weights["self_correction"] * c +
            weights["accuracy"] * a
        )
        combined.append(score)
    
    return combined
```

---

## 8. Evaluation Framework

### 8.1 Pass@1 Accuracy

**File**: `eval/benchmark.py`

**What it measures**: Fraction of problems solved correctly.

```python
def pass_at_1(generations, gold):
    """generations: list of generated texts; gold: str."""
    correct = 0
    for gen in generations:
        pred = extract_answer(gen)
        try:
            from math_verify import parse, verify
            ok = verify(parse(str(gold)), parse(pred))
        except:
            ok = pred == str(gold)
        if ok:
            correct += 1
    return correct / len(generations)
```

**Protocol**:
- Sample 64 completions per problem
- Calculate mean accuracy
- Report with confidence intervals

### 8.2 Claim-Level Reliability (CLR)

**File**: `eval/clr_eval.py`

**What it measures**: Reasoning reliability, not just answer accuracy.

**Process**:
1. Sample K=32 traces per problem
2. Extract M=5 claims from each trace
3. Self-verify each claim (ask model true/false)
4. Cluster traces by extracted answer
5. Pick answer with highest reliability sum

**Why CLR matters**: A model might get the right answer but with flawed reasoning. CLR detects this.

### 8.3 Format Compliance

**What it measures**: Percentage of outputs with proper `<thinking>` and `<answer>` tags.

```python
def format_compliance(outputs):
    compliant = 0
    for output in outputs:
        if ("<thinking>" in output and "</thinking>" in output and
            "<answer>" in output and "</answer>" in output):
            compliant += 1
    return compliant / len(outputs)
```

### 8.4 Self-Correction Rate

**What it measures**: Percentage of complex problems with genuine backtracking.

```python
def self_correction_rate(outputs, difficulty="hard"):
    corrected = 0
    for output in outputs:
        if has_backtracking(output):
            corrected += 1
    return corrected / len(outputs)
```

---

## 9. Model Export & Serving

### 9.1 LoRA Merge

**File**: `export/merge_lora.py`

**Purpose**: Combine LoRA adapters with base model for deployment.

```python
# Load model
model, tokenizer = FastLanguageModel.from_pretrained(model_path)

# Merge LoRA weights
model = model.merge_and_unload()

# Save merged model
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
```

**Output**: Full model in FP16 (~8GB for 4B model)

### 9.2 GGUF Quantization

**File**: `export/quantize_gguf.py`

**Purpose**: Convert to GGUF format for llama.cpp/Ollama.

**Quantization levels**:
| Method | Bits | Size | Quality |
|--------|------|------|---------|
| F16 | 16 | ~8GB | Perfect |
| Q6_K | 6 | ~3.5GB | Near-perfect |
| Q5_K_M | 5 | ~3.0GB | Very good |
| Q4_K_M | 4 | ~2.5GB | Good |

### 9.3 vLLM Serving

**File**: `export/serve_vllm.py`

**Purpose**: High-throughput inference server.

```bash
# Start server
python export/serve_vllm.py --model export/merged --port 8000

# Test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "self-correcting-reasoner", "messages": [{"role": "user", "content": "What is 2+3*4?"}]}'
```

### 9.4 Ollama Integration

**File**: `export/serve_ollama.py`

**Purpose**: Local serving with Ollama.

```bash
# Create model
python export/serve_ollama.py \
  --model-path export/gguf/model-q5_k_m.gguf \
  --action create

# Run
ollama run self-correcting-reasoner
```

---

## 10. Code Walkthrough

### 10.1 Data Preparation (`data/gsm8k_prep.py`)

```python
def extract_answer(answer_text):
    """Extract numerical answer from GSM8K format."""
    # GSM8K answers end with #### <number>
    match = re.search(r"####\s*(.+)$", answer_text)
    if match:
        return match.group(1).strip()
    return answer_text.strip()

def create_reasoning_trace(question, solution):
    """Create structured reasoning trace."""
    answer = extract_answer(solution)
    reasoning = re.sub(r"####\s*.+$", "", solution, flags=re.MULTILINE).strip()
    
    trace = f"""<thinking>
{reasoning}
</thinking>
<answer>
{answer}
</answer>"""
    
    return trace
```

### 10.2 SFT Training (`train/sft_trainer.py`)

Key steps:
1. Check CUDA availability
2. Load model with Unsloth/PEFT
3. Prepare dataset with chat template
4. Configure SFTTrainer
5. Train with response-only masking
6. Save checkpoints

### 10.3 GRPO Training (`train/grpo_trainer.py`)

Key steps:
1. Load SFT checkpoint
2. Prepare dataset (prompts + gold answers)
3. Define reward function
4. Configure GRPOTrainer
5. Train with group sampling
6. Monitor rewards and KL divergence

### 10.4 Evaluation (`eval/benchmark.py`)

Key steps:
1. Load test dataset
2. Sample N generations per problem
3. Extract answers from tags
4. Calculate Pass@1 accuracy
5. Save results to JSON

---

## 11. Configuration Reference

### 11.1 SFT Configuration

```yaml
# configs/sft_config.yaml
base_model: "Qwen/Qwen3-4B"
sequence_len: 2048
micro_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 0.0002
epochs: 3
optim: "adamw_bnb_8bit"
bf16: true
lora_r: 16
lora_alpha: 16
```

### 11.2 GRPO Configuration

```yaml
# configs/grpo_config.yaml
model_path: "runs/sft"
num_generations: 8
beta: 0.1
learning_rate: 0.00001
epochs: 1
reward_weights:
  format: 0.3
  self_correction: 0.3
  accuracy: 0.4
```

---

## 12. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA OOM | Batch too large | Reduce batch_size, increase grad_accum |
| Low accuracy | Insufficient training | Increase epochs or data |
| No self-correction | Reward too weak | Increase self_correction weight |
| Format non-compliance | Reward too weak | Increase format weight |
| GRPO instability | Learning rate too high | Reduce lr to 1e-6 |
| Quantization degradation | Too aggressive | Use Q5_K_M instead of Q4_K_M |

### VRAM Estimates

| Model Size | FP16 | 4-bit QLoRA |
|------------|------|-------------|
| 4B | 8GB | 2.5GB |
| 7B | 14GB | 4GB |
| 13B | 26GB | 7GB |

---

## References

1. **QLoRA**: Dettmers et al. "QLoRA: Efficient Finetuning of Quantized Language Models" (2023)
2. **GRPO**: Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (2024)
3. **LoRA**: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
4. **GSM8K**: Cobbe et al. "Training Verifiers to Solve Math Word Problems" (2021)
5. **MATH**: Hendrycks et al. "Measuring Mathematical Problem Solving With the MATH Dataset" (2021)
6. **Unsloth**: https://github.com/unslothai/unsloth
7. **TRL**: https://github.com/huggingface/trl
