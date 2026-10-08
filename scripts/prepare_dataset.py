import argparse
import json
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train.jsonl")
    parser.add_argument("--output", default="data/train")
    args = parser.parse_args()

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item.get("messages"), list):
                raise ValueError(f"{args.input}:{n}: messages must be a list")
            rows.append(item)

    Dataset.from_list(rows).save_to_disk(args.output)
    print(f"Saved {len(rows)} examples to {args.output}")

if __name__ == "__main__":
    main()
