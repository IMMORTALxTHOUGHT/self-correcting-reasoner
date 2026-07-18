#!/usr/bin/env python3
"""
Prepare ONLY the held-out test splits for evaluation.

Unlike prepare_all.py (which downloads ~80-100K examples across 6 sources and
takes a long time), this pulls just the small GSM8K-test and MATH-test splits
and writes them to data/processed/. Use it to generate an eval set quickly
without re-processing all training data.

Outputs:
  data/processed/gsm8k_test.jsonl
  data/processed/math_test.jsonl

Each record uses the same message/metadata format as prepare_all.py so the
trainers and eval scripts are consistent.
"""

import json
import re
from pathlib import Path

from datasets import load_dataset

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _gsm8k_records(split):
    records = []
    for item in split:
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
    return records


def _math_records(split):
    records = []
    for item in split:
        ans_match = re.search(r"\\boxed\{(.*?)\}", item["solution"], re.DOTALL)
        answer = ans_match.group(1).strip() if ans_match else ""
        records.append({
            "messages": [
                {"role": "user", "content": item["problem"]},
                {"role": "assistant", "content": f"<thinking>\n{item['solution']}\n</thinking>\n<answer>\n{answer}\n</answer>"}
            ],
            "metadata": {"dataset": "math", "level": item.get("level", ""), "answer": answer}
        })
    return records


def _save(records, filename):
    out = PROCESSED_DIR / filename
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {len(records)} examples -> {out}")
    return records


def prepare_gsm8k_test():
    print("[GSM8K test]")
    ds = load_dataset("openai/gsm8k", "main")
    recs = _gsm8k_records(ds["test"])
    return _save(recs, "gsm8k_test.jsonl")


def prepare_math_test():
    print("[MATH test]")
    try:
        ds = load_dataset("qwedsacf/competition_math")
    except Exception as e:
        print(f"  MATH unavailable ({e}); skipping.")
        return []
    if "test" not in ds:
        print("  no 'test' split in dataset; skipping.")
        return []
    recs = _math_records(ds["test"])
    return _save(recs, "math_test.jsonl")


if __name__ == "__main__":
    prepare_gsm8k_test()
    prepare_math_test()
    print("\nDone. Held-out test splits written under data/processed/.")
