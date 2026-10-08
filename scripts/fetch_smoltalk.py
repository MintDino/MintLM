import argparse
import json
from pathlib import Path

from datasets import load_dataset


def write_split(dataset_name, config, split, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_name, config, split=split, streaming=True)

    written = 0
    skipped = 0

    with output.open("w", encoding="utf-8") as f:
        for row in ds:
            messages = row.get("messages")

            if not isinstance(messages, list) or not messages:
                skipped += 1
                continue

            valid = True
            normalized = []

            for message in messages:
                if not isinstance(message, dict):
                    valid = False
                    break

                role = message.get("role")
                content = message.get("content")

                if role not in {"system", "user", "assistant"}:
                    valid = False
                    break

                if not isinstance(content, str) or not content.strip():
                    valid = False
                    break

                normalized.append({
                    "role": role,
                    "content": content,
                })

            if not valid:
                skipped += 1
                continue

            json.dump({"messages": normalized}, f, ensure_ascii=False)
            f.write("\n")
            written += 1

            if written % 10000 == 0:
                print(f"{split}: {written:,} examples written")

    print(f"{split}: wrote {written:,} examples to {output}")
    if skipped:
        print(f"{split}: skipped {skipped:,} invalid rows")


def main():
    parser = argparse.ArgumentParser(
        description="Download the complete HuggingFaceTB/smoltalk dataset into MintLM JSONL."
    )
    parser.add_argument("--dataset", default="HuggingFaceTB/smoltalk")
    parser.add_argument("--config", default="all")
    parser.add_argument("--train-output", default="data/smoltalk/train.jsonl")
    parser.add_argument("--valid-output", default="data/smoltalk/valid.jsonl")
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    print(f"Config:  {args.config}")
    print("Using the complete train split and the complete test split as validation.")
    print("Streaming is enabled so the full dataset does not need to fit in RAM.")

    write_split(args.dataset, args.config, "train", args.train_output)
    write_split(args.dataset, args.config, "test", args.valid_output)


if __name__ == "__main__":
    main()
