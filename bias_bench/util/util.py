# def _is_generative(model):
#     # Checks if we are running an autoregressive model.
#     return model in [
#         "GPT2LMHeadModel",
#         "SentenceDebiasGPT2LMHeadModel",
#         "INLPGPT2LMHeadModel",
#         "CDAGPT2LMHeadModel",
#         "DropoutGPT2LMHeadModel",
#         "SelfDebiasGPT2LMHeadModel",
#     ]

def start_token_mapper(model_name):
    start_token_mapper = {
        "gpt2": "<|endoftext|>",
        "llama": "<s>",
        "phi": "<|endoftext|>",
    }
    return start_token_mapper[model_name]

def _is_generative(model):
    # Checks if we are running an autoregressive model.
    return model in [
        "GPT2LMHeadModel",
        "SentenceDebiasGPT2LMHeadModel",
        "INLPGPT2LMHeadModel",
        "CDAGPT2LMHeadModel",
        "DropoutGPT2LMHeadModel",
        "SelfDebiasGPT2LMHeadModel",
        "LlamaForCausalLM", # For llama 2 models
        "PhiForCausalLM", # For phi models
        "AutoModelForCausalLM", # For other huggingface models
    ]


def _is_self_debias(model):
    # Checks if we are running a Self-Debias model.
    return model in [
        "SelfDebiasGPT2LMHeadModel",
        "SelfDebiasBertForMaskedLM",
        "SelfDebiasAlbertForMaskedLM",
        "SelfDebiasRobertaForMaskedLM",
    ]
