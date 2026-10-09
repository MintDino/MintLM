# MintLM

MintLM is a QLoRA fine-tuning project for building a custom assistant on top of Qwen3-8B.

## Stack

- Base: `unsloth/Qwen3-8B-Base-unsloth-bnb-4bit`
- Fine-tuning: QLoRA
- Trainer: TRL SFTTrainer
- Accelerator: Unsloth
- Dataset: Hugging Face `HuggingFaceTB/smoltalk`

## Google Colab training

A step-by-step Colab setup for the Tesla T4, with conservative VRAM settings, Google Drive persistence, automatic checkpoint resume, and a short smoke test is in [`colab/README.md`](colab/README.md).

## Dataset

MintLM uses the **complete SmolTalk dataset**, not the tiny starter dataset.

SmolTalk is an approximately 1M-example supervised fine-tuning dataset created by Hugging Face for the SmolLM2-Instruct family. Its `all` configuration contains the complete mix documented by the dataset authors, including instruction-following, rewriting, summarization, mathematics, coding, system-prompt, long-context, conversation, and function-calling data.

Source: https://huggingface.co/datasets/HuggingFaceTB/smoltalk

The dataset is downloaded at training-machine setup time and is intentionally **not committed to GitHub**.

### Download the complete dataset

```bash
python scripts/fetch_smoltalk.py
```

This streams the complete `all/train` split into `data/smoltalk/train.jsonl` and the complete `all/test` split into `data/smoltalk/valid.jsonl`.

Streaming keeps the complete dataset from being loaded into RAM at once.

The Hugging Face dataset contains multiple Parquet shards and is several gigabytes on disk, so make sure the Codespace/GPU machine has enough storage.

### Train

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/fetch_smoltalk.py

python scripts/train.py --config configs/qwen3-8b-qlora.yaml
```

You can override the dataset paths if needed:

```bash
python scripts/train.py \
  --train-file data/smoltalk/train.jsonl \
  --valid-file data/smoltalk/valid.jsonl \
  --config configs/qwen3-8b-qlora.yaml
```

## Goals

MintLM is being tuned toward programming, Minecraft development, practical technical problem solving, and direct conversational behavior.

The base SmolTalk mixture already contains dedicated data for mathematics, coding, system-prompt following, long-context understanding, rewriting, summarization, conversations, and function calling. MintLM's first serious training run therefore uses the complete published mixture rather than arbitrarily replacing or sampling away parts of it.

## Data format

Each JSONL record is a chat conversation:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

The downloader validates roles and non-empty message content while preserving the original conversations.

Only train on data you have permission to use. Do not put private conversations, secrets, personal data, copyrighted books, or scraped personal information into the dataset.

## Inference

```bash
python inference/chat.py --adapter outputs/mintlm-qwen3-8b
```

## Roadmap

- [x] QLoRA training skeleton
- [x] Chat dataset format
- [x] Local inference
- [x] Complete SmolTalk dataset pipeline
- [x] Automated evaluation
- [ ] Preference tuning
- [ ] GGUF export
- [ ] Hugging Face release
