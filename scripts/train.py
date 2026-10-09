import unsloth
import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer


QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def main():
    parser = argparse.ArgumentParser(description="Train MintLM with QLoRA.")
    parser.add_argument("--config", default="configs/qwen3-8b-qlora.yaml")
    parser.add_argument("--train-file", default="data/smoltalk/train.jsonl")
    parser.add_argument("--valid-file", default="data/smoltalk/valid.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--resume",
        choices=["auto", "none"],
        default="auto",
        help="Automatically resume from the latest checkpoint in output-dir, or start fresh.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional session limit. -1 means finish the configured epoch(s).",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. In Colab, select Runtime > Change runtime type > T4 GPU.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output/checkpoints: {output_dir.resolve()}")
    print("Loading the 4-bit Qwen3 base model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        dtype=None,
    )

    if not tokenizer.chat_template:
        print("Tokenizer has no chat template; applying the standard Qwen3 conversation format.")
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    for path in (args.train_file, args.valid_file):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {path}\n"
                "Run scripts/fetch_smoltalk.py first, or pass the correct --train-file/--valid-file."
            )

    print("Loading JSONL datasets (this can take a while on the first run)...")
    raw = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.valid_file},
    )

    def format_row(row):
        return {
            "text": tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    train_ds = raw["train"].map(
        format_row,
        remove_columns=raw["train"].column_names,
        desc="Formatting training conversations",
    )
    valid_ds = raw["validation"]
    eval_limit = int(cfg.get("max_eval_samples", 512))
    if eval_limit > 0 and len(valid_ds) > eval_limit:
        valid_ds = valid_ds.select(range(eval_limit))
    valid_ds = valid_ds.map(
        format_row,
        remove_columns=valid_ds.column_names,
        desc="Formatting validation conversations",
    )

    bf16 = torch.cuda.is_bf16_supported()
    train_args_values = {
        "output_dir": str(output_dir),
        "num_train_epochs": cfg["num_train_epochs"],
        "per_device_train_batch_size": cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": cfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "learning_rate": cfg["learning_rate"],
        "warmup_ratio": cfg["warmup_ratio"],
        "weight_decay": cfg["weight_decay"],
        "logging_steps": cfg["logging_steps"],
        "eval_strategy": "steps",
        "eval_steps": cfg["eval_steps"],
        "save_strategy": "steps",
        "save_steps": cfg["save_steps"],
        "save_total_limit": cfg["save_total_limit"],
        "seed": cfg["seed"],
        "bf16": bf16,
        "fp16": not bf16,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "adamw_8bit",
        "lr_scheduler_type": "cosine",
        "report_to": "none",
        "dataset_text_field": "text",
        "max_length": cfg["max_seq_length"],
        "packing": False,
        "save_only_model": False,
        "dataloader_num_workers": 2,
        "remove_unused_columns": True,
    }
    if args.max_steps is not None:
        train_args_values["max_steps"] = args.max_steps
    elif "max_steps" in cfg:
        train_args_values["max_steps"] = cfg["max_steps"]

    train_args = SFTConfig(**train_args_values)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        args=train_args,
    )

    resume_path = None
    if args.resume == "auto":
        resume_path = get_last_checkpoint(str(output_dir))
        if resume_path:
            print(f"Resuming from checkpoint: {resume_path}")
        else:
            print("No checkpoint found; starting a new training run.")

    print(f"Training examples: {len(train_ds):,}")
    print(f"Validation examples used per evaluation: {len(valid_ds):,}")
    print(f"Epochs: {cfg['num_train_epochs']}; sequence length: {cfg['max_seq_length']}")
    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved final adapter and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
