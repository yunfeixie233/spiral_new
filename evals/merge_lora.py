import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into a base model and push to HuggingFace"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="HuggingFace path to the base model",
    )
    parser.add_argument(
        "--lora-path", type=str, required=True, help="Path to the LoRA adapter weights"
    )
    parser.add_argument(
        "--target-path",
        type=str,
        required=True,
        help="HuggingFace path to save the merged model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for merging (cuda, cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model loading and saving",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading base model from {args.base_model}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map=args.device, trust_remote_code=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"Loading LoRA adapter from {args.lora_path}")
    # Load and merge LoRA weights
    model = PeftModel.from_pretrained(model, args.lora_path)
    print("Merging LoRA weights with base model")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.target_path}")
    # Save the merged model
    model.push_to_hub(args.target_path)
    tokenizer.push_to_hub(args.target_path)

    print("Merge and save completed successfully!")


if __name__ == "__main__":
    main()
