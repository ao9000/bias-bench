import argparse
import os
import json

import torch
import transformers

from bias_bench.benchmark.crows import CrowSPairsRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id, _is_generative, _is_self_debias

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs CrowS-Pairs benchmark.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="SentenceDebiasBertForMaskedLM",
    choices=[
        "SentenceDebiasBertForMaskedLM",
        "SentenceDebiasAlbertForMaskedLM",
        "SentenceDebiasRobertaForMaskedLM",
        "SentenceDebiasGPT2LMHeadModel",
        
        "SentenceDebiasPhi2LMHeadModel", # For phi 2 debiased models
        "SentenceDebiasLlama2LMHeadModel", # For llama 2 debiased models

        "INLPBertForMaskedLM",
        "INLPAlbertForMaskedLM",
        "INLPRobertaForMaskedLM",
        "INLPGPT2LMHeadModel",

        "INLPPhi2LMHeadModel", # For phi 2 debiased models
        "INLPLlama2LMHeadModel", # For llama 2 debiased models

        "CDABertForMaskedLM",
        "CDAAlbertForMaskedLM",
        "CDARobertaForMaskedLM",
        "CDAGPT2LMHeadModel",
        
        "CDAPhi2LMHeadModel", # For phi 2 debiased models
        "CDALlama2LMHeadModel", # For llama 2 debiased models

        "DropoutBertForMaskedLM",
        "DropoutAlbertForMaskedLM",
        "DropoutRobertaForMaskedLM",
        "DropoutGPT2LMHeadModel",

        "SelfDebiasBertForMaskedLM",
        "SelfDebiasAlbertForMaskedLM",
        "SelfDebiasRobertaForMaskedLM",
        "SelfDebiasGPT2LMHeadModel",

        "SelfDebiasLlama2LMHeadModel", # For llama 2 debiased models
        "SelfDebiasPhi2LMHeadModel", # For phi 2 debiased models

        "DropoutLlama2LMHeadModel", # For llama 2 debiased models
        "DropoutPhi2LMHeadModel", # For phi 2 debiased models
    ],
    help="Model to evalute (e.g., SentenceDebiasBertForMaskedLM). Typically, these "
    "correspond to a HuggingFace class.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    choices=["bert-base-uncased", "albert-base-v2", "roberta-base", "gpt2", "microsoft/phi-2", "meta-llama/Llama-2-7b-hf"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--bias_direction",
    action="store",
    type=str,
    help="Path to the file containing the pre-computed bias direction for SentenceDebias.",
)
parser.add_argument(
    "--projection_matrix",
    action="store",
    type=str,
    help="Path to the file containing the pre-computed projection matrix for INLP.",
)
parser.add_argument(
    "--load_path",
    action="store",
    type=str,
    help="Path to saved ContextDebias, CDA, or Dropout model checkpoint.",
)
parser.add_argument(
   "--bias_type",
   action="store",
   default=None,
   choices=["race-color", "gender", "socioeconomic", "sexual-orientation", "religion", "age", "nationality", "disability", "physical-appearance"],
   help="Determines which CrowS-Pairs dataset split to evaluate against.",
)
parser.add_argument(
   "--ckpt_num",
   action="store",
   type=int,
    default=None,
   help="Checkpoint number to be included in the experiment ID and results file name.",
)


def get_debias_method():
    if "SelfDebias".lower() in args.model.lower():
        return "SelfDebias"
    elif "INLP".lower() in args.model.lower():
        return "INLP"
    elif "CDA".lower() in args.model.lower():
        return "CDA"
    elif "Dropout".lower() in args.model.lower():
        return "Dropout"
    elif "SentenceDebias".lower() in args.model.lower():
        return "SentenceDebias"


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="crows",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        bias_type=args.bias_type,
    )

    print("Running CrowS-Pairs benchmark:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - bias_direction: {args.bias_direction}")
    print(f" - projection_matrix: {args.projection_matrix}")
    print(f" - load_path: {args.load_path}")
    print(f" - bias_type: {args.bias_type}")

    kwargs = {}
    if args.bias_direction is not None:
        # Load the pre-computed bias direction for SentenceDebias.
        bias_direction = torch.load(args.bias_direction)
        kwargs["bias_direction"] = bias_direction

    if args.projection_matrix is not None:
        # Load the pre-computed projection matrix for INLP.
        projection_matrix = torch.load(args.projection_matrix)
        kwargs["projection_matrix"] = projection_matrix

    # Load model and tokenizer. `load_path` can be used to override `model_name_or_path`.
    model = getattr(models, args.model)(
        args.load_path or args.model_name_or_path, **kwargs
    )

    if _is_self_debias(args.model):
        model._model.eval()
    else:
        model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    runner = CrowSPairsRunner(
        model=model,
        tokenizer=tokenizer,
        input_file=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized.csv",
        bias_type=args.bias_type,
        is_generative=_is_generative(args.model),  # Affects model scoring.
        is_self_debias=_is_self_debias(args.model),
        model_name_or_path=args.model_name_or_path, # Added to determine unconditional start token
    )
    results = runner()
    print(f"Metric: {results}")

    # Modified
    # Remove any slash from file experiment_id
    experiment_id = experiment_id.replace("/", "_")

    os.makedirs(f"{args.persistent_dir}/results/crows", exist_ok=True)
    
    file_path = f"{args.persistent_dir}/results/crows_{get_debias_method()}/{experiment_id}.json"
    if args.ckpt_num is not None:
        # Add checkpoint number to results as key
        if os.path.exists(file_path):
            # Change to append mode
            existing_results = {}
            with open(file_path, "r") as f:
                # Load prev res
                existing_results = json.load(f)
            # Add new res to existing res
            existing_results[str(args.ckpt_num)] = results
            results = existing_results
        else:
            # First run
            results = {args.ckpt_num: results}

    with open(file_path, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)
        # json.dump(results, f)