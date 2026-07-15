#!/usr/bin/env python3
"""
Unified data preparation — all sources, capped for 24GB single-GPU training.

Target: ~80-100K examples total (manageable for Qwen3.5-2B on A5000 24GB)

Sources:
- GSM8K (~7.5K) — foundational math
- MATH (~12.5K) — competition math
- OpenMathInstruct-2 (~50K sample) — large-scale math traces
- Fable-5-traces (~4.6K) — agent reasoning
- Claude Mythos 25K (~5K filtered) — reasoning/science/planning
- Reasoning Summaries 61K (~10K filtered) — mixed reasoning

Output: data/processed/{source}.jsonl + data/processed/all_train.jsonl
"""

import json
import random
import re
import sys
from pathlib import Path
from datasets import load_dataset

SEED = 42
random.seed(SEED)
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Max examples per source (tune down if GPU OOM or too slow)
MAX_GSM8K = 8000
MAX_MATH = 15000
MAX_OPENMATH = 50000
MAX_FABLE5 = 5000
MAX_MYTHOS = 5000
MAX_REASONING = 10000


def save_jsonl(records, filename):
    out = PROCESSED_DIR / filename
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {len(records)} examples -> {out}")
    return records


# ============================================================================
# GSM8K
# ============================================================================

def prepare_gsm8k():
    print("[GSM8K]")
    ds = load_dataset("openai/gsm8k", "main")
    records = []
    for split_name in ["train", "test"]:
        for item in ds[split_name]:
            ans_match = re.search(r"####\s*(.+)$", item["answer"])
            answer = ans_match.group(1).strip() if ans_match else ""
            reasoning = re.sub(r"####\s*.+$", "", item["answer"], flags=re.MULTILINE).strip()
            records.append({
                "messages": [
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": f"<thinking>\n{reasoning}\n</thinking>\n<answer>\n{answer}\n</answer>"}
                ],
                "metadata": {"dataset": "gsm8k", "answer": answer}
            })
    records = records[:MAX_GSM8K]
    return save_jsonl(records, "gsm8k.jsonl")


# ============================================================================
# MATH (competition_math)
# ============================================================================

def prepare_math():
    print("[MATH]")
    try:
        ds = load_dataset("hendrycks/competition_math", "main")
    except Exception as e:
        print(f"  Failed: {e}")
        return []
    records = []
    for split_name in ["train", "test"]:
        if split_name not in ds:
            continue
        for item in ds[split_name]:
            ans_match = re.search(r"\\boxed\{(.*?)\}", item["solution"], re.DOTALL)
            answer = ans_match.group(1).strip() if ans_match else ""
            records.append({
                "messages": [
                    {"role": "user", "content": item["problem"]},
                    {"role": "assistant", "content": f"<thinking>\n{item['solution']}\n</thinking>\n<answer>\n{answer}\n</answer>"}
                ],
                "metadata": {"dataset": "math", "level": item.get("level", ""), "answer": answer}
            })
    records = records[:MAX_MATH]
    return save_jsonl(records, "math.jsonl")


# ============================================================================
# OpenMathInstruct-2 (sample)
# ============================================================================

def prepare_openmath():
    print("[OpenMathInstruct-2]")
    try:
        ds = load_dataset("nvidia/OpenMathInstruct-2", split="train")
    except Exception as e:
        print(f"  Failed: {e}")
        return []

    total = len(ds)
    indices = random.sample(range(total), min(MAX_OPENMATH, total))
    records = []

    for idx in indices:
        item = ds[idx]
        problem = item.get("problem", "")
        solution = item.get("solution", "")

        if not problem or not solution:
            continue

        # Extract boxed answer
        ans_match = re.search(r"\\boxed\{(.*?)\}", solution, re.DOTALL)
        answer = ans_match.group(1).strip() if ans_match else ""

        records.append({
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": f"<thinking>\n{solution}\n</thinking>\n<answer>\n{answer}\n</answer>"}
            ],
            "metadata": {"dataset": "openmathinstruct2"}
        })

    return save_jsonl(records, "openmathinstruct2.jsonl")


# ============================================================================
# Fable-5 Traces
# ============================================================================

def prepare_fable5():
    print("[Fable5]")
    try:
        ds = load_dataset("Glint-Research/Fable-5-traces", split="train")
    except Exception as e:
        print(f"  Failed: {e}")
        return []

    records = []
    for item in ds:
        context = item.get("context", "")
        cot = item.get("cot", "")
        completion = item.get("completion", "")
        output_type = item.get("output_type", "")

        if not completion and not cot:
            continue

        thinking = cot if cot else ""
        response = completion if completion else ""
        user_msg = context[-2000:] if len(context) > 2000 else context

        records.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"<thinking>\n{thinking}\n</thinking>\n<answer>\n{response}\n</answer>"}
            ],
            "metadata": {"dataset": "fable5", "output_type": output_type}
        })

    records = records[:MAX_FABLE5]
    return save_jsonl(records, "fable5.jsonl")


# ============================================================================
# Claude Mythos Distilled 25K (filtered to math/science/planning)
# ============================================================================

def prepare_mythos():
    print("[Mythos]")
    try:
        ds = load_dataset("WithinUsAI/claude_mythos_distilled_25k", split="train")
    except Exception as e:
        print(f"  Failed: {e}")
        return []

    keep_categories = {"mathematical_reasoning", "scientific_analysis", "agentic_planning"}
    records = []

    for item in ds:
        messages = item.get("messages", [])
        category = item.get("category", "")

        if len(messages) < 2 or category not in keep_categories:
            continue

        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        if not user_msg or not assistant_msg:
            continue

        records.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"<thinking>\n{assistant_msg}\n</thinking>\n<answer>\n(distilled from {category})\n</answer>"}
            ],
            "metadata": {"dataset": "mythos", "category": category}
        })

    records = records[:MAX_MYTHOS]
    return save_jsonl(records, "mythos.jsonl")


# ============================================================================
# Reasoning Summaries 61K (filtered to math/code)
# ============================================================================

def prepare_reasoning():
    print("[Reasoning Summaries]")
    try:
        ds = load_dataset("SupraLabs/reasoning-summaries-61k", split="train")
    except Exception as e:
        print(f"  Failed: {e}")
        return []

    math_kw = re.compile(
        r"equation|proof|theorem|solve|calculate|integral|derivative|matrix|"
        r"vector|probability|algorithm|function|def |class |return |import ",
        re.IGNORECASE
    )

    records = []
    for item in ds:
        user_msg = item.get("user", "")
        assistant_msg = item.get("assistant", "")

        if not user_msg or not assistant_msg:
            continue

        combined = user_msg + " " + assistant_msg
        if not math_kw.search(combined):
            continue

        records.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"<thinking>\n{assistant_msg}\n</thinking>\n<answer>\n(extracted from reasoning-summaries)\n</answer>"}
            ],
            "metadata": {"dataset": "reasoning_summaries"}
        })

    records = random.sample(records, min(MAX_REASONING, len(records)))
    return save_jsonl(records, "reasoning_summaries.jsonl")


# ============================================================================
# Merge
# ============================================================================

def merge_all():
    all_records = []
    for f in sorted(PROCESSED_DIR.glob("*.jsonl")):
        if f.name == "all_train.jsonl":
            continue
        with open(f) as fp:
            for line in fp:
                if line.strip():
                    all_records.append(line)

    random.shuffle(all_records)
    out = PROCESSED_DIR / "all_train.jsonl"
    with open(out, "w") as f:
        f.writelines(all_records)

    print(f"\n{'='*50}")
    print(f"MERGED: {len(all_records)} total examples -> {out}")
    print(f"{'='*50}")

    # Print breakdown
    counts = {}
    for f in sorted(PROCESSED_DIR.glob("*.jsonl")):
        if f.name == "all_train.jsonl":
            continue
        with open(f) as fp:
            c = sum(1 for _ in fp)
        counts[f.stem] = c
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name:25s} {count:>6,}")
    print(f"  {'TOTAL':25s} {sum(counts.values()):>6,}")

    return len(all_records)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    sources = sys.argv[1:] if len(sys.argv) > 1 else [
        "gsm8k", "math", "openmath", "fable5", "mythos", "reasoning", "merge"
    ]

    if "gsm8k" in sources: prepare_gsm8k()
    if "math" in sources: prepare_math()
    if "openmath" in sources: prepare_openmath()
    if "fable5" in sources: prepare_fable5()
    if "mythos" in sources: prepare_mythos()
    if "reasoning" in sources: prepare_reasoning()
    if "merge" in sources: merge_all()
