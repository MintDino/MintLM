import argparse
import torch
from unsloth import FastLanguageModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    messages = [{
        "role": "system",
        "content": "You are MintLM, a helpful technical assistant. Be direct, accurate, and practical.",
    }]

    print("MintLM ready. Type /exit to quit.")

    while True:
        user = input("\nYou: ").strip()
        if user == "/exit":
            break
        if not user:
            continue

        messages.append({"role": "user", "content": user})
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output = model.generate(
                input_ids=inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                do_sample=True,
            )

        answer = tokenizer.decode(
            output[0][inputs.shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        print(f"MintLM: {answer}")
        messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
