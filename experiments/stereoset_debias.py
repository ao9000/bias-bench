import argparse
import json
import os

import torch
import transformers

from bias_bench.benchmark.stereoset import StereoSetRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id, _is_generative, _is_self_debias


thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs StereoSet benchmark.")
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
    default="SentenceDebiasForMaskedLM",
    choices=[
        "SentenceDebiasBertForMaskedLM",
        "SentenceDebiasAlbertForMaskedLM",
        "SentenceDebiasRobertaForMaskedLM",
        "SentenceDebiasGPT2LMHeadModel",

        "SentenceDebiasPhi2LMHeadModel", # Added for SentenceDebias Phi model
        "SentenceDebiasLlama2LMHeadModel", # Added for SentenceDebias Llama model

        "INLPBertForMaskedLM",
        "INLPAlbertForMaskedLM",
        "INLPRobertaForMaskedLM",
        "INLPGPT2LMHeadModel",

        "CDABertForMaskedLM",
        "CDAAlbertForMaskedLM",
        "CDARobertaForMaskedLM",
        "CDAGPT2LMHeadModel",

        "DropoutBertForMaskedLM",
        "DropoutAlbertForMaskedLM",
        "DropoutRobertaForMaskedLM",
        "DropoutGPT2LMHeadModel",

        "SelfDebiasGPT2LMHeadModel",
        "SelfDebiasBertForMaskedLM",
        "SelfDebiasAlbertForMaskedLM",
        "SelfDebiasRobertaForMaskedLM",

        "SelfDebiasLlama2LMHeadModel", # Added for self debiasing Llama model
        "SelfDebiasPhi2LMHeadModel", # Added for self debiasing Phi model

        "INLPPhi2LMHeadModel",  # For phi 2 debiased models
        "INLPLlama2LMHeadModel",  # For llama 2 debiased models
    ],
    help="Model to evalute (e.g., SentenceDebiasBertForMaskedLM).",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    choices=["bert-base-uncased",
             "albert-base-v2",
             "roberta-base",
             "gpt2",
             "microsoft/phi-2",
             "meta-llama/Llama-2-7b-hf"],
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
    help="Path to saved CDA or Dropout model checkpoint.",
)
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=1,
    help="The batch size to use during StereoSet intrasentence evaluation.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    type=str,
    choices=["race-color", "gender", "socioeconomic", "sexual-orientation", "religion", "age", "nationality", "disability", "physical-appearance"],
    default=None,
    help="The type of bias to mitigate.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=None,
    help="RNG seed. Used for logging in experiment ID.",
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
        name="stereoset",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        bias_type=args.bias_type,
        seed=args.seed,
    )

    print("Running StereoSet:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - bias_direction: {args.bias_direction}")
    print(f" - projection_matrix: {args.projection_matrix}")
    print(f" - load_path: {args.load_path}")
    print(f" - batch_size: {args.batch_size}")
    print(f" - bias_type: {args.bias_type}")
    print(f" - seed: {args.seed}")

    kwargs = {}
    if args.bias_direction is not None:
        # Load the pre-computed bias direction for SentenceDebias.
        bias_direction = torch.load(args.bias_direction)
        kwargs["bias_direction"] = bias_direction
        print("Loaded bias direction.")

    if args.projection_matrix is not None:
        # Load the pre-computed projection matrix for INLP.
        projection_matrix = torch.load(args.projection_matrix)
        kwargs["projection_matrix"] = projection_matrix

    model = getattr(models, args.model)(
        args.load_path or args.model_name_or_path, **kwargs
    )

    if _is_self_debias(args.model):
        model._model.eval()
    else:
        model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Use self-debiasing name.
    bias_type = args.bias_type
    if bias_type == "race":
        bias_type = "race-color"

    runner = StereoSetRunner(
        intrasentence_model=model,
        tokenizer=tokenizer,
        input_file=f"{args.persistent_dir}/data/stereoset/test.json",
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        is_generative=_is_generative(args.model),
        is_self_debias=_is_self_debias(args.model),
        bias_type=bias_type,
    )
    results = runner()

    # Remove any slash from file experiment_id
    experiment_id = experiment_id.replace("/", "_")

    os.makedirs(f"{args.persistent_dir}/results/stereoset", exist_ok=True)
    with open(
        f"{args.persistent_dir}/results/stereoset_{get_debias_method()}/{experiment_id}.json", "w"
    ) as f:
        json.dump(results, f, indent=2)
