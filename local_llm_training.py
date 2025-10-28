#!/usr/bin/env python3
"""
Local LLM Training Script

Train a small causal language model (e.g., GPT-2) on popular datasets
like ELI5, SQuAD, or WikiText with checkpointing and a final export.

Usage examples (see README for more):

python local_llm_training.py \
    --model gpt2 \
    --dataset eli5 \
    --num-samples 10000 \
    --epochs 3 \
    --batch-size 4 \
    --output-dir ./my_trained_model
"""

import os
import json
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from rich.console import Console
from rich.panel import Panel

console = Console()


def ensure_dirs(base_out: str) -> Dict[str, str]:
    os.makedirs(base_out, exist_ok=True)
    paths = {
        "checkpoints": os.path.join(base_out, "checkpoints"),
        "logs": os.path.join(base_out, "logs"),
        "final": os.path.join(base_out, "final_model"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def load_and_prepare_dataset(name: str, num_samples: int, max_length: int, custom_data_file: Optional[str] = None) -> Dataset:
    """Load dataset by name and return a Dataset with a single 'text' column."""
    console.print(f"[cyan]Loading dataset: {name} (samples={num_samples})[/cyan]")

    if name == "custom":
        if not custom_data_file or not os.path.exists(custom_data_file):
            raise ValueError("--custom-data-file must be provided and exist for custom dataset")
        with open(custom_data_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        texts = texts[:num_samples]
        return Dataset.from_dict({"text": texts})

    if name == "eli5":
        ds = load_dataset("eli5", split=f"train[:{num_samples}]")

        def fmt(example):
            title = example.get("title") or example.get("question_title") or ""
            ans = example.get("selftext")
            if not ans:
                answers = example.get("answers")
                if isinstance(answers, list) and answers:
                    first = answers[0]
                    if isinstance(first, dict):
                        ans = first.get("text") or ""
                    else:
                        ans = str(first)
            text = f"Question: {title}\nAnswer: {ans or ''}"
            return {"text": text}

        ds = ds.map(fmt, remove_columns=ds.column_names)
        # Filter out empty texts
        ds = ds.filter(lambda ex: bool(ex.get("text")) and len(ex["text"].strip()) > 0)
        return ds

    if name == "squad":
        ds = load_dataset("squad", split=f"train[:{num_samples}]")

        def fmt(example):
            question = example.get("question", "")
            answers = example.get("answers", {})
            answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
            answer = answer_texts[0] if answer_texts else example.get("context", "")
            text = f"Question: {question}\nAnswer: {answer}"
            return {"text": text}

        ds = ds.map(fmt, remove_columns=ds.column_names)
        ds = ds.filter(lambda ex: bool(ex.get("text")) and len(ex["text"].strip()) > 0)
        return ds

    if name == "wikitext":
        # Load full wikitext-2 raw train split, then filter and downselect
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # Filter out empty lines before mapping
        ds = ds.filter(lambda ex: bool(ex.get("text")) and len(ex["text"].strip()) > 0)

        def fmt(example):
            return {"text": example.get("text", "").strip()}

        ds = ds.map(fmt, remove_columns=ds.column_names)
        # Filter again to ensure non-empty strings
        ds = ds.filter(lambda ex: bool(ex.get("text")) and len(ex["text"].strip()) > 0)

        # Downselect to requested number of samples
        if num_samples and len(ds) > num_samples:
            ds = ds.select(range(num_samples))
        return ds

    raise ValueError(f"Unsupported dataset: {name}")


def train(args: argparse.Namespace) -> None:
    console.print(Panel.fit("[bold cyan]Starting Training[/bold cyan]", style="cyan"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        console.print("[green]✓ Using GPU[/green]")
    else:
        console.print("[yellow]⚠ Using CPU (training will be slow)[/yellow]")

    # Prepare output directories
    dirs = ensure_dirs(args.output_dir)

    # Load tokenizer and model
    console.print(f"[cyan]Loading model: {args.model}[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Load dataset
    dataset = load_and_prepare_dataset(args.dataset, args.num_samples, args.max_length, args.custom_data_file)

    def make_tokenized(max_len: int):
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_len,
                return_attention_mask=True,
            )

        tk = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        return tk.filter(lambda ex: ex.get("input_ids") is not None and len(ex["input_ids"]) > 0)

    tokenized = make_tokenized(args.max_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def latest_checkpoint(path: str) -> Optional[str]:
        if not os.path.exists(path):
            return None
        try:
            subdirs = [d for d in os.listdir(path) if d.startswith("checkpoint-")]
            if not subdirs:
                return None
            steps = []
            for d in subdirs:
                try:
                    n = int(d.split("checkpoint-")[-1])
                    steps.append((n, d))
                except Exception:
                    continue
            if not steps:
                return None
            steps.sort(key=lambda x: x[0])
            return os.path.join(path, steps[-1][1])
        except Exception:
            return None

    # Save a copy of the config
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({**vars(args), "started_at": datetime.utcnow().isoformat()}, f, indent=2)

    # Train with automatic OOM fallback
    resume_path = args.resume_from or latest_checkpoint(dirs["checkpoints"]) 
    cur_bs = args.batch_size
    cur_max_len = args.max_length

    while True:
        training_args = TrainingArguments(
            output_dir=dirs["checkpoints"],
            num_train_epochs=args.epochs,
            per_device_train_batch_size=cur_bs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_steps=500,
            save_total_limit=5,
            logging_dir=dirs["logs"],
            logging_steps=50,
            learning_rate=args.learning_rate,
            fp16=torch.cuda.is_available(),
            bf16=False,
            report_to=[],
            tf32=True if torch.cuda.is_available() else False,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            optim="adamw_torch",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )

        console.print(f"[cyan]Training... (batch_size={cur_bs}, max_length={cur_max_len})[/cyan]")
        try:
            trainer.train(resume_from_checkpoint=resume_path)
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda error: out of memory" in msg:
                console.print("[yellow]CUDA OOM encountered. Adjusting settings and resuming...[/yellow]")
                if cur_bs > 1:
                    cur_bs = max(1, cur_bs // 2)
                else:
                    # As a last resort, reduce sequence length and re-tokenize
                    if cur_max_len > 128:
                        cur_max_len = max(96, cur_max_len // 2)
                        tokenized = make_tokenized(cur_max_len)
                    else:
                        raise
                resume_path = latest_checkpoint(dirs["checkpoints"]) or resume_path
                continue
            else:
                raise

    console.print("[green]✓ Training complete, saving final model[/green]")
    model.save_pretrained(dirs["final"]) 
    tokenizer.save_pretrained(dirs["final"]) 

    console.print(f"[bold green]Final model saved to: {dirs['final']}[/bold green]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local LLM Training")
    parser.add_argument("--model", type=str, default="gpt2", help="Base model to fine-tune")
    parser.add_argument("--dataset", type=str, default="eli5", choices=["eli5", "squad", "wikitext", "custom"], help="Dataset to train on")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples to use")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--output-dir", type=str, default="./my_trained_model", help="Output directory for checkpoints and final model")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length for tokenization")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--custom-data-file", type=str, default=None, help="Path to a custom text file (one example per line)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)