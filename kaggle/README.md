# MintLM — Kaggle Training

This is the free-GPU training path for MintLM.

## Hardware

Use **Kaggle Notebook → T4 x2**. Kaggle currently lists T4 x2 with 2 × 16 GB GPUs and up to 12 hours per notebook execution. GPU availability is quota/queue dependent.

## Dataset

Only the complete `HuggingFaceTB/smoltalk` dataset is used.

The `all` config contains the complete SmolTalk training split plus its test split. The dataset is about 2.11 GB of Parquet files for the `all` config.

## Kaggle setup

Create a new Kaggle Notebook and set:

**Settings → Accelerator → T4 x2**

Then turn **Internet ON**.

Run these cells in order.

### Cell 1 — clone MintLM

```bash
!git clone https://github.com/MintDino/MintLM.git /kaggle/working/MintLM
%cd /kaggle/working/MintLM
```

### Cell 2 — install dependencies

```bash
!pip install -q -U unsloth transformers trl datasets accelerate peft bitsandbytes pyyaml
```

### Cell 3 — verify the GPUs

```bash
!nvidia-smi
```

You should see two T4 GPUs.

### Cell 4 — download the complete SmolTalk dataset

```bash
!python scripts/fetch_smoltalk.py \
    --dataset HuggingFaceTB/smoltalk \
    --config all \
    --train-output /kaggle/working/smoltalk/train.jsonl \
    --valid-output /kaggle/working/smoltalk/valid.jsonl
```

This downloads the complete train split and complete test split. The generated JSONL is outside the Git repository.

### Cell 5 — train

For the first run, use one GPU to validate the pipeline:

```bash
!CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config configs/qwen3-8b-qlora.yaml \
    --train-file /kaggle/working/smoltalk/train.jsonl \
    --valid-file /kaggle/working/smoltalk/valid.jsonl
```

If that starts successfully, stop it and switch to the multi-GPU launcher below.

### Cell 6 — two-GPU training

```bash
!torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/qwen3-8b-qlora.yaml \
    --train-file /kaggle/working/smoltalk/train.jsonl \
    --valid-file /kaggle/working/smoltalk/valid.jsonl
```

**Important:** if the current Unsloth/TRL version reports a distributed-training error, use the one-GPU command instead. The model fits the T4-class setup more easily with QLoRA than full fine-tuning.

## Saving your progress

Kaggle's working disk is not the same thing as GitHub. Before the session ends, use Kaggle's **Save Version → Save & Run All** for notebook reproducibility.

For long training, the next improvement is to push checkpoints to a Hugging Face model repository so a new Kaggle session can resume without relying on the previous VM.

Do not commit the dataset or model checkpoints to GitHub.

## Recommended first run

Do **not** immediately burn a full GPU session.

First confirm:

1. both GPUs are visible;
2. SmolTalk downloads successfully;
3. Qwen3-8B loads in 4-bit;
4. one training step completes;
5. a checkpoint is written.

Then run the long training job in repeated Kaggle sessions.

## Expected storage

The `all` SmolTalk Parquet data is about 2.11 GB. The converted JSONL will be larger, and the 4-bit model/checkpoints also consume storage, so keep an eye on `/kaggle/working` and delete failed/old checkpoints when necessary.
