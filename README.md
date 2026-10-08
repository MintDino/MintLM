# MintLM

MintLM is a QLoRA fine-tuning project for building a custom assistant on top of Qwen3-8B.

## Stack

- Base: `unsloth/Qwen3-8B-Base-unsloth-bnb-4bit`
- Fine-tuning: QLoRA
- Trainer: TRL SFTTrainer
- Accelerator: Unsloth
- Dataset: JSONL chat conversations

## Goals

MintLM is being tuned toward programming, Minecraft development, practical technical problem solving, and direct conversational behavior.

## Training

Use a CUDA GPU. A cloud GPU is recommended for Qwen3-8B.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --config configs/qwen3-8b-qlora.yaml
```

The repository contains a tiny starter dataset only. Replace it with a properly licensed, high-quality dataset before serious training.

## Dataset

Each line in `data/*.jsonl` is:

```json
{"messages":[{"role":"system","content":"You are MintLM."},{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

Only train on data you have permission to use. Do not put private conversations, secrets, personal data, copyrighted books, or scraped personal information into the dataset.

## Inference

```bash
python inference/chat.py --adapter outputs/mintlm-qwen3-8b
```

## Roadmap

- [x] QLoRA training skeleton
- [x] Chat dataset format
- [x] Local inference
- [ ] Large curated dataset
- [x] Automated evaluation
- [ ] Preference tuning
- [ ] GGUF export
- [ ] Hugging Face release
