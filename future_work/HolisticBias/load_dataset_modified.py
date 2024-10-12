# Reference from HolisticBiasTeacher class
# Link: https://github.com/facebookresearch/ResponsibleNLP/blob/main/holistic_bias/run_bias_calculation.py

import os
import csv
import pandas as pd


def load_and_filter_data(data_folder):
    # Load sentences.csv
    sentences_file = os.path.join(data_folder, "sentences.csv")
    sentences_df = pd.read_csv(sentences_file)

    # Load noun_phrases.csv
    noun_phrases_file = os.path.join(data_folder, "noun_phrases.csv")
    noun_phrases_df = pd.read_csv(noun_phrases_file)

    # Filter sentences
    filtered_sentences = sentences_df[
        (sentences_df["noun_phrase_type"].isin(["descriptor_noun", "noun_descriptor"]))
        & (sentences_df["descriptor_gender"] == "(none)")
    ]

    print(f"{len(filtered_sentences)} valid sentences identified.")

    return filtered_sentences

