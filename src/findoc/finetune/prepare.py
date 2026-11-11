"""Build chat-format JSONL training data from FinQA, using the same prompt as production.

    python -m findoc.finetune.prepare --out data/finetune --negatives 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from ..qa import NOT_FOUND, build_messages

DATASET = "ChanceFocus/flare-finqa"
SPLITS = {"train": "train", "valid": "valid", "test": "test"}
LABEL = "filing excerpt"

_QUERY_RE = re.compile(r"Context:\s*(?P<context>.*)\nQuestion:\s*(?P<question>.*?)\s*\nAnswer:\s*$", re.S)


def parse_query(query: str) -> tuple[str, str] | None:
    """Split a FinQA prompt into its context and question."""
    match = _QUERY_RE.search(query)
    if not match:
        return None
    return match["context"].strip(), match["question"].strip()


def to_example(question: str, context: str, answer: str) -> dict:
    messages = build_messages(question, [(LABEL, context)])
    return {"messages": messages + [{"role": "assistant", "content": answer}]}


def convert(rows: list[dict], negatives: float = 0.0, seed: int = 13) -> list[dict]:
    """Convert rows, adding unanswerable examples that teach the model to refuse.

    A negative pairs a question with a different filing's context. Without these, every
    training example has an answer, which teaches the model to always produce a figure.
    """
    rng = random.Random(seed)
    parsed = []
    for row in rows:
        pair = parse_query(row["query"])
        if pair:
            parsed.append((pair[0], pair[1], str(row["answer"]).strip()))

    examples = [
        to_example(question, context, f"{answer} [1]") for context, question, answer in parsed
    ]
    if len(parsed) > 1:
        for _ in range(int(len(parsed) * negatives)):
            first, second = rng.sample(parsed, 2)
            if first[0] != second[0]:
                examples.append(to_example(first[1], second[0], NOT_FOUND))
    rng.shuffle(examples)
    return examples


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/finetune"))
    parser.add_argument("--negatives", type=float, default=0.1, help="share of refusal examples")
    parser.add_argument("--limit", type=int, help="cap rows per split, for smoke tests")
    args = parser.parse_args()

    from datasets import load_dataset

    for name, split in SPLITS.items():
        rows = list(load_dataset(DATASET, split=split))
        if args.limit:
            rows = rows[: args.limit]
        # Refusals go in every split so evaluation can measure them as well.
        examples = convert(rows, negatives=args.negatives)
        path = args.out / f"{name}.jsonl"
        write_jsonl(examples, path)
        print(f"{path}: {len(examples)} examples from {len(rows)} rows")


if __name__ == "__main__":
    main()
