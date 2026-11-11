"""Compare the base model with a trained adapter on the same held-out examples.

    python -m findoc.finetune.evaluate --model Qwen/Qwen2.5-7B-Instruct \
        --adapter outputs/qwen-finqa --test data/finetune/test.jsonl --limit 200
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

from ..llm import TransformersLLM
from ..qa import NOT_FOUND

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_CITATION_RE = re.compile(r"\[\d+\]")


def _numbers(text: str) -> list[float]:
    return [float(match.replace(",", "")) for match in _NUMBER_RE.findall(text)]


def is_correct(prediction: str, gold: str, tolerance: float = 0.005) -> bool:
    """Score one answer. For numeric gold, the prediction's final number must match.

    Using the final number stops a long answer from scoring by merely restating a figure.
    FinQA mixes 0.12 and 12%, so the two scales count as the same answer.
    """
    gold = _CITATION_RE.sub("", gold).strip()
    prediction = _CITATION_RE.sub("", prediction).strip()
    if gold == NOT_FOUND:
        return NOT_FOUND.lower().rstrip(".") in prediction.lower()
    gold_numbers = _numbers(gold)
    if not gold_numbers:
        return prediction.lower().startswith(gold.lower())
    predicted = _numbers(prediction)
    if not predicted:
        return False
    target, value = abs(gold_numbers[-1]), abs(predicted[-1])
    allowed = max(target * tolerance, 1e-9)
    return any(abs(value * scale - target) <= allowed for scale in (1, 100, 0.01))


def evaluate(llm, examples: list[dict], max_new_tokens: int) -> dict:
    totals = {
        "answerable": 0, "answerable_correct": 0,
        "unanswerable": 0, "unanswerable_refused": 0,
        "cited": 0,
    }
    predictions = []
    for example in examples:
        messages, gold = example["messages"][:-1], example["messages"][-1]["content"]
        prediction = llm.generate(messages, max_new_tokens=max_new_tokens)
        correct = is_correct(prediction, gold)
        if gold == NOT_FOUND:
            totals["unanswerable"] += 1
            totals["unanswerable_refused"] += correct
        else:
            totals["answerable"] += 1
            totals["answerable_correct"] += correct
            totals["cited"] += bool(_CITATION_RE.search(prediction))
        predictions.append({"gold": gold, "prediction": prediction, "correct": correct})
    summary = {
        "accuracy": totals["answerable_correct"] / max(totals["answerable"], 1),
        "refusal_rate": totals["unanswerable_refused"] / max(totals["unanswerable"], 1),
        "citation_rate": totals["cited"] / max(totals["answerable"], 1),
        **totals,
    }
    return {"summary": summary, "predictions": predictions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", help="omit to evaluate only the base model")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--load-in-4bit", action="store_true")
    # Generous on purpose: the base model works step by step, and a cut-off answer would score
    # as wrong, flattering the adapter.
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("outputs/eval.json"))
    args = parser.parse_args()

    with args.test.open(encoding="utf-8") as handle:
        examples = [json.loads(line) for line in handle][: args.limit]

    results = {}
    runs = [("base", "")] + ([("adapter", args.adapter)] if args.adapter else [])
    for name, adapter in runs:
        llm = TransformersLLM(args.model, adapter, args.load_in_4bit)
        results[name] = evaluate(llm, examples, args.max_new_tokens)
        print(name, json.dumps(results[name]["summary"]))
        del llm
        gc.collect()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
