import argparse
from peft import AutoPeftModelForCausalLM
import torch
import os


def main(adapters_path):
    model = AutoPeftModelForCausalLM.from_pretrained(adapters_path, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload(progressbar=True)

    output_merged_dir = os.path.join(adapters_path, "merged_adapters")
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    print("Saved final merged checkpoint to:", output_merged_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and save a PEFT adapter to a pre-trained model.")
    parser.add_argument('--adapter_model_path', type=str, required=True, help='Path to trained adapters')
    args = parser.parse_args()

    main(args.adapter_model_path)
