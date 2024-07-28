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

# use this code to find out how much tokens are in the prefix for each model
# from transformers import AutoTokenizer
# def get_self_debias_prefix_token_count(model_name):
#     DEBIASING_PREFIXES = {
#         "race-color": "The following text discriminates against people because of their race/color: ",
#         "gender": "The following text discriminates against people because of their gender: ",
#         "socioeconomic": "The following text discriminates against people because of their socioeconomic status/occupation: ",
#         "sexual-orientation": "The following text discriminates against people because of their sexual orientiation: ",
#         "religion": "The following text discriminates against people because of their religion: ",
#         "age": "The following text discriminates against people because of their age: ",
#         "nationality": "The following text discriminates against people because of their nationality: ",
#         "disability": "The following text discriminates against people because of their disability: ",
#         "physical-appearance": "The following text discriminates against people because of their physical appearance: ",
#     }
#     # Encode the prefixes
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     prefix_token_counts = {}
#
#     for bias_type, prefix in DEBIASING_PREFIXES.items():
#         input_ids = tokenizer.encode(prefix, return_tensors="pt")
#         prefix_token_counts[bias_type] = len(input_ids[0])
#         print(f"Model: {model_name}, Bias Type: {bias_type}, Prefix Token Count: {len(input_ids[0])}")
#
#     return prefix_token_counts

def get_target_modules_for_model(model_name):
    # Return all linear layers to be as good as full fine-tuning performance
    target_modules = {
        "gpt2": ["c_attn", "c_proj", "c_fc"],
        "microsoft/phi-2": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        "meta-llama/Llama-2-7b-hf": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }

    return target_modules[model_name]


# This code is the calculated prefix token count for each model, to save loading time
def get_self_debias_prefix_token_count(model_name):
    self_debias_token_count = {"gpt2":
    {
        'race-color': 15,
         'gender': 13,
         'socioeconomic': 17,
         'sexual-orientation': 15,
         'religion': 13,
         'age': 13,
         'nationality': 13,
         'disability': 13,
         'physical-appearance': 14,
    },
    "microsoft/phi-2":
    {
        'race-color': 15,
         'gender': 13,
         'socioeconomic': 17,
         'sexual-orientation': 15,
         'religion': 13,
         'age': 13,
         'nationality': 13,
         'disability': 13,
         'physical-appearance': 14
    },
        "meta-llama/Llama-2-7b-hf":
    {
        'race-color': 17,
         'gender': 15,
         'socioeconomic': 22,
         'sexual-orientation': 17,
         'religion': 15,
         'age': 15,
         'nationality': 16,
         'disability': 16,
         'physical-appearance': 16}
    }

    return self_debias_token_count[model_name]


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
        "GPT2LMHeadModel_NonBFloat16" # For sentence debiasing and INLP
        "SentenceDebiasGPT2LMHeadModel",
        "INLPGPT2LMHeadModel",
        "CDAGPT2LMHeadModel",
        "DropoutGPT2LMHeadModel", # For dropout models
        "SelfDebiasGPT2LMHeadModel",

        "AutoModelForCausalLM",  # For other huggingface models
        "LlamaForCausalLM", # For llama 2 models
        "PhiForCausalLM",  # For phi models

        "SelfDebiasLlama2LMHeadModel", # For llama 2 models (Self-Debias)
        "SelfDebiasPhi2LMHeadModel", # For phi 2 models (Self-Debias)

        "CDAPhi2LMHeadModel", # For phi models (CDA)
        "CDALlama2LMHeadModel", # For llama 2 models (CDA)

        "SentenceDebiasPhi2LMHeadModel", # For phi 2 models (SentenceDebias)
        "SentenceDebiasLlama2LMHeadModel", # For llama 2 models (SentenceDebias)

        "INLPPhi2LMHeadModel", # For phi 2 models (INLP)
        "INLPLlama2LMHeadModel", # For llama 2 models (INLP)

        "PhiForCausalLM_NonBFloat16", # For phi models
        "LlamaForCausalLM_NonBFloat16", # For llama 2 models

        "DropoutLlama2LMHeadModel", # For dropout models (Llama 2)
        "DropoutPhi2LMHeadModel", # For dropout models (Phi 2)

    ]


def _is_self_debias(model):
    # Checks if we are running a Self-Debias model.
    return model in [
        "SelfDebiasGPT2LMHeadModel",
        "SelfDebiasBertForMaskedLM",
        "SelfDebiasAlbertForMaskedLM",
        "SelfDebiasRobertaForMaskedLM",
        
        "SelfDebiasPhi2LMHeadModel", # For phi 2 models
        "SelfDebiasLlama2LMHeadModel", # For llama 2 models
    ]
