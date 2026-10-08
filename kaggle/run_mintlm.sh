#!/bin/bash
set -euo pipefail

cd /kaggle/working/MintLM

echo "=== GPU ==="
nvidia-smi

echo "=== Installing dependencies ==="
pip install -q -U unsloth transformers trl datasets accelerate peft bitsandbytes pyyaml

echo "=== Downloading complete SmolTalk ==="
python scripts/fetch_smoltalk.py \
  --dataset HuggingFaceTB/smoltalk \
  --config all \
  --train-output /kaggle/working/smoltalk/train.jsonl \
  --valid-output /kaggle/working/smoltalk/valid.jsonl

echo "=== Starting MintLM training ==="
python scripts/train.py \
  --config configs/qwen3-8b-qlora.yaml \
  --train-file /kaggle/working/smoltalk/train.jsonl \
  --valid-file /kaggle/working/smoltalk/valid.jsonl
