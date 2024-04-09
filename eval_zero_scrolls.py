import json
import os
from pathlib import Path
import sys
from datetime import datetime
import random

import numpy as np
import torch
from datasets import load_dataset
from fire import Fire
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed as hf_set_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

datasets = [
    "gov_report",
    "summ_screen_fd",
    "qmsum",
    "qasper",
    "narrative_qa",
    "quality",
    "musique",
    "squality",
    "space_digest",
    "book_sum_sort",
]

model_to_max_input_tokens = {
    "sheared-llama-1.3b": 4096,
}


def trim_doc_keeping_suffix(tokenizer, tokenized_input_full, example, suffix_index, max_tokens, device):
    seperator_and_suffix = f"{example['truncation_seperator'].strip()}\n\n{example['input'][suffix_index:].strip()}\n"
    tokenized_seperator_and_suffix = tokenizer(seperator_and_suffix, return_tensors="pt").input_ids.to(device)
    tokenized_input_trimmed = tokenized_input_full[:, : max_tokens - tokenized_seperator_and_suffix.shape[1]]
    tokenized_input = torch.cat([tokenized_input_trimmed, tokenized_seperator_and_suffix], dim=1)
    return tokenized_input


def process_model_input(tokenizer, example, max_tokens, device):
    tokenized_input_full = tokenizer(example["input"], return_tensors="pt").input_ids.to(device)
    if tokenized_input_full.shape[1] <= max_tokens:
        return tokenized_input_full

    seperator_and_query_text = example["truncation_seperator"] + example["input"][example["query_start_index"] :]
    tokenized_seperator_and_query = tokenizer(seperator_and_query_text, return_tensors="pt").input_ids.to(device)
    input_without_query = example["input"][: example["query_start_index"]]
    tokenized_input_without_query = tokenizer(input_without_query, return_tensors="pt").input_ids.to(device)
    tokenized_input_without_query = tokenized_input_without_query[
        :, : max_tokens - tokenized_seperator_and_query.shape[1]
    ]

    tokenized_input = torch.cat([tokenized_input_without_query, tokenized_seperator_and_query], dim=1)
    return tokenized_input


def main(
    model_name="/home/louchao/vqtree/data/sheared-llama-1.3b", generations_dir="generations", max_examples_per_task=-1
):
    seed = 43
    random.seed(seed)
    np.random.seed(seed)
    hf_set_seed(seed)
    print("Params:")
    print(f"model: {model_name}")
    _model_name = Path(model_name).name
    generations_dir = os.path.join(generations_dir, _model_name.replace("-", "_"))
    print(f"generations_dir: {generations_dir}")
    print(f"max_examples_per_task: {max_examples_per_task}")
    print("=" * 50)
    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    print(f"time as start: {time}")

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_new_tokens = 1024  # default length defined by the benchmark
    max_input_length = model_to_max_input_tokens[_model_name] - max_new_tokens

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    model = model.eval()

    print(f"{model} model loaded!, device:{model.device}")

    print("Will write to:", generations_dir)
    os.makedirs(generations_dir, exist_ok=True)
    for dataset in datasets:
        generations = dict()
        print(f"Processing {dataset}")
        time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        print(f"time as start {dataset}: {time}")
        print(f"Loading {dataset}")
        data = load_dataset("/home/louchao/vqtree/data/zero_scrolls", dataset, trust_remote_code=True)
        print(f"Loaded {dataset}")

        for i, example in enumerate(data["test"]):

            if 0 < max_examples_per_task == i:
                print(f"Reached {max_examples_per_task} for {dataset}. Breaking")
                break

            model_input = process_model_input(tokenizer, example, max_input_length, device)

            prediction_token_ids = model.generate(
                model_input, max_new_tokens=max_new_tokens, do_sample=False, temperature=1
            )

            predicted_text = tokenizer.decode(prediction_token_ids[0], skip_special_tokens=True)
            generations[example["id"]] = predicted_text

        out_file_path = os.path.join(generations_dir, f"preds_{dataset}.json")
        with open(out_file_path, "w") as f_out:
            json.dump(generations, f_out, indent=4)

        print(f"Done generating {len(generations)} examples from {dataset}")
    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    print(f"time at end: {time}")
    print(f"Look for predictions in {generations_dir}")


if __name__ == "__main__":
    Fire(main)
