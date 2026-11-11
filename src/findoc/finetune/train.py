"""Train a LoRA adapter, in 4-bit (QLoRA) on a GPU or full precision (LoRA) elsewhere.

    python -m findoc.finetune.train --train data/finetune/train.jsonl \
        --eval data/finetune/valid.jsonl --output outputs/qwen-finqa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# These attention projections are named the same in Qwen and Llama.
DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj"


def encode_example(tokenizer, messages: list[dict], max_length: int) -> dict | None:
    """Tokenize one example, masking the prompt so loss covers only the answer.

    Returns None when it does not fit. Truncating would cut off the answer, or the evidence
    the answer depends on, so an example that long is dropped instead.
    """
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(messages[-1]["content"] + tokenizer.eos_token, add_special_tokens=False)[
        "input_ids"
    ]
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_length:
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def load_examples(path: Path, tokenizer, max_length: int, limit: int | None = None) -> list[dict]:
    encoded, dropped = [], 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if limit and len(encoded) >= limit:
                break
            example = encode_example(tokenizer, json.loads(line)["messages"], max_length)
            if example is None:
                dropped += 1
            else:
                encoded.append(example)
    print(f"{path}: {len(encoded)} examples, {dropped} dropped for exceeding {max_length} tokens")
    if not encoded:
        raise ValueError(f"No usable examples in {path}. Raise --max-length?")
    return encoded


class PadCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        longest = max(len(item["input_ids"]) for item in batch)
        padding = {"input_ids": self.pad_token_id, "attention_mask": 0, "labels": -100}
        return {
            key: torch.tensor(
                [item[key] + [fill] * (longest - len(item[key])) for item in batch],
                dtype=torch.long,
            )
            for key, fill in padding.items()
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["auto", "qlora", "lora"], default="auto")
    parser.add_argument("--target-modules", default=DEFAULT_TARGETS)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="overrides --epochs when set")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, help="cap examples per file, for smoke tests")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    has_gpu = torch.cuda.is_available()
    mode = args.mode if args.mode != "auto" else ("qlora" if has_gpu else "lora")
    if mode == "qlora" and not has_gpu:
        raise SystemExit("QLoRA needs a CUDA GPU, and none was found. Use --mode lora or Colab.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data = load_examples(args.train, tokenizer, args.max_length, args.limit)
    eval_data = (
        load_examples(args.eval, tokenizer, args.max_length, args.limit) if args.eval else None
    )

    use_bf16 = has_gpu and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    load_kwargs: dict = {}
    if mode == "qlora":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        load_kwargs["device_map"] = {"": 0}
    elif has_gpu:
        load_kwargs["dtype"] = compute_dtype
        load_kwargs["device_map"] = {"": 0}
    else:
        load_kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.config.use_cache = False
    if mode == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif has_gpu:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.r,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=args.target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=0.05,  # a float below 1 is a share of total steps
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch" if eval_data else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=bool(eval_data),
        metric_for_best_model="eval_loss" if eval_data else None,
        greater_is_better=False if eval_data else None,
        bf16=use_bf16,
        fp16=has_gpu and not use_bf16,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    result = trainer.train()

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    metadata = {
        "base_model": args.model,
        "mode": mode,
        "train_examples": len(train_data),
        "train_loss": result.training_loss,
        "steps": result.global_step,
    }
    (args.output / "findoc_training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved adapter to {args.output}: {metadata}")


if __name__ == "__main__":
    main()
