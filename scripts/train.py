import argparse
import torch
import yaml
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3-8b-qlora.yaml")
    parser.add_argument("--train-file", default="data/train.jsonl")
    parser.add_argument("--valid-file", default="data/valid.jsonl")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for QLoRA training.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        dtype=None,
    )

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

    raw = load_dataset("json", data_files={
        "train": args.train_file,
        "validation": args.valid_file,
    })

    def format_row(row):
        return {
            "text": tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    train_ds = raw["train"].map(format_row, remove_columns=raw["train"].column_names)
    valid_ds = raw["validation"].map(format_row, remove_columns=raw["validation"].column_names)

    bf16 = torch.cuda.is_bf16_supported()

    train_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        seed=cfg["seed"],
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        report_to="none",
        dataset_text_field="text",
        max_length=cfg["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        args=train_args,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Saved adapter to {cfg['output_dir']}")

if __name__ == "__main__":
    main()
