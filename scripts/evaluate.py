import argparse
import math
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--file", default="data/valid.jsonl")
    parser.add_argument("--max-examples", type=int, default=50)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    model.eval()

    dataset = load_dataset("json", data_files=args.file)["train"]
    dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    losses = []
    with torch.inference_mode():
        for row in dataset:
            text = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(model.device)
            result = model(**inputs, labels=inputs["input_ids"])
            losses.append(float(result.loss))

    mean_loss = sum(losses) / len(losses)
    print(json.dumps({
        "examples": len(losses),
        "mean_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
    }, indent=2))

if __name__ == "__main__":
    main()
