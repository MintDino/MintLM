# MintLM — Google Colab (Tesla T4 / 16 GB)

This workflow fine-tunes `unsloth/Qwen3-8B-Base-unsloth-bnb-4bit` with QLoRA on the complete `HuggingFaceTB/smoltalk` `all` configuration. A full epoch over the entire dataset is a long-running job; free Colab sessions are not guaranteed to last long enough. Checkpoints and data are kept in Google Drive so a new session can resume.

## Before you start

In Colab choose **Runtime → Change runtime type → T4 GPU**. Run the cells below in order. If Colab restarts, rerun the setup cells and the training cell; training automatically resumes from the latest checkpoint in Drive.

### Cell 1 — Mount Drive and set paths

```python
from google.colab import drive
drive.mount("/content/drive")

import os
from pathlib import Path

REPO_DIR = Path("/content/MintLM")
DRIVE_DIR = Path("/content/drive/MyDrive/MintLM")
DATA_DIR = DRIVE_DIR / "data" / "smoltalk"
OUTPUT_DIR = DRIVE_DIR / "checkpoints" / "mintlm-qwen3-8b"

DRIVE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

### Cell 2 — Clone or update the repository

```python
!if [ -d /content/MintLM/.git ]; then git -C /content/MintLM pull; else git clone https://github.com/MintDino/MintLM.git /content/MintLM; fi
%cd /content/MintLM
```

### Cell 3 — Install dependencies

```python
%cd /content/MintLM
!pip install -q -U unsloth transformers trl datasets accelerate peft bitsandbytes pyyaml
```

If Colab requires a runtime restart after installation, restart it and rerun Cells 1–3.

### Cell 4 — Check GPU

```python
!nvidia-smi
```

### Cell 5 — Download the complete SmolTalk dataset to Drive

This may take a while and uses several GB of Drive storage. It streams the Hugging Face source and writes JSONL; the dataset is not added to Git.

```python
%cd /content/MintLM
train_file = str(DATA_DIR / "train.jsonl")
valid_file = str(DATA_DIR / "valid.jsonl")

if not os.path.exists(train_file) or not os.path.exists(valid_file):
    !python scripts/fetch_smoltalk.py --config all --train-output "{train_file}" --valid-output "{valid_file}"
else:
    print("SmolTalk JSONL files already exist in Drive; skipping download.")
```

### Cell 6 — Copy dataset to local runtime storage (faster training I/O)

```python
import shutil
LOCAL_DATA = Path("/content/smoltalk")
LOCAL_DATA.mkdir(parents=True, exist_ok=True)
local_train = LOCAL_DATA / "train.jsonl"
local_valid = LOCAL_DATA / "valid.jsonl"

if not local_train.exists():
    shutil.copy2(train_file, local_train)
if not local_valid.exists():
    shutil.copy2(valid_file, local_valid)

print("Train bytes:", local_train.stat().st_size)
print("Validation bytes:", local_valid.stat().st_size)
```

### Cell 7 — Start or resume training

The checkpoint directory is on Drive. The script automatically finds the latest `checkpoint-*` folder and resumes optimizer, scheduler, and model state. Do not delete checkpoint folders while training.

```python
%cd /content/MintLM
!python scripts/train.py \
  --config configs/qwen3-8b-qlora.yaml \
  --train-file /content/smoltalk/train.jsonl \
  --valid-file /content/smoltalk/valid.jsonl \
  --output-dir "/content/drive/MyDrive/MintLM/checkpoints/mintlm-qwen3-8b"
```

### Optional — limit a test run

Before committing to the full dataset, test the pipeline for 10 optimizer steps:

```python
!python scripts/train.py \
  --config configs/qwen3-8b-qlora.yaml \
  --train-file /content/smoltalk/train.jsonl \
  --valid-file /content/smoltalk/valid.jsonl \
  --output-dir "/content/drive/MyDrive/MintLM/checkpoints/mintlm-qwen3-8b-smoke-test" \
  --resume none \
  --max-steps 10
```

The smoke test uses a separate output directory so it won't overwrite the main run. When it succeeds, run the full training cell. With `max_steps=-1`, the configured epoch covers the complete valid training split; it is not capped to a small subset.

## Important notes

- T4 has 16 GB VRAM; settings are conservative (4-bit weights, sequence length 2048, micro-batch 1, gradient accumulation 16, gradient checkpointing, 8-bit optimizer). If an out-of-memory error occurs, reduce `max_seq_length` to 1024 in the YAML and restart the runtime.
- A single epoch over ~1M examples can take far longer than one free Colab session. Re-run Cell 7 in a new session to continue from the Drive checkpoint.
- Drive is the durable copy; `/content` is temporary. Never store the only checkpoint in `/content`.
- The complete test split is saved as validation, but each evaluation uses at most 512 examples to avoid evaluating the entire test set every 500 steps.
- Do not use the smoke-test output directory as the full training output.
