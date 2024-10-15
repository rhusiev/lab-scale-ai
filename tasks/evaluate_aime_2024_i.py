from collections.abc import Sequence
import evaluate
import numpy as np
import json
import argparse
import torch
import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel
from typing import Iterable
from tqdm import tqdm
from os import path, makedirs, getenv

from generate_from_hf_model import generate_from_prompt

from collections import Counter
import re
import string
from typing import Optional, List

#####
# TODO: Below is partially adapted better answer parsing from
# https://github.com/vlievin/medical-reasoning/blob/master/medical_reasoning/models/functional/infer_answer.py
# to be completed, and new metric added.


def parse_options_from_input(input_question: str) -> dict:
    # extract the options part from the input question
    options_str = re.search(r"\{(.+?)\}$", input_question)
    if options_str:
        options_str = options_str.group(1)
        options = dict(item.split(": ") for item in options_str.split(", "))
        return options
    else:
        return {}


def get_start_indices(target: str, pattern: str) -> list[int]:
    try:
        matches = re.finditer(pattern, target)
        return [m.start() for m in matches]
    except Exception as exc:
        return []


def get_first_match(query, choices, keys, op=min):
    assert len(choices) == len(keys)
    indices = [(key, get_start_indices(query, o)) for key, o in zip(keys, choices)]
    indices = list(filter(lambda x: len(x[1]), indices))
    if len(indices):
        return op(indices, key=lambda x: x[1])[0]
    else:
        return None


def infer_answer_from_input(input_question: str, target_answer: str) -> Optional[str]:
    options = parse_options_from_input(input_question)
    if not options:
        return None

    # check if the target answer is directly one of the option keys
    if target_answer.strip() in options:
        return target_answer.strip()

    # direct match with the provided options' values
    for key, value in options.items():
        if value.strip() == target_answer.strip():
            return key

    # use regex patterns to match the answer
    option_symbols = list(options.keys())
    option_values = list(options.values())
    option_symbols_re = [rf"{re.escape(o)}(\)|:|\.|,| )" for o in option_symbols]

    # try to match using option symbols
    match = get_first_match(target_answer, option_symbols_re, option_symbols)
    if match is not None:
        return match

    # try to match using the full text of the options
    match = get_first_match(target_answer, option_values, option_symbols)
    if match is not None:
        return match

    return None


###########
# Following code from SQUAD, here:
# https://github.com/huggingface/transformers/blob/main/src/transformers/data/metrics/squad_metrics.py


def normalize_answer(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(int(normalize_answer(a_gold)) == int(normalize_answer(a_pred)))


############


def evaluate_hf_model_aime(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data: Sequence[dict[str, str]],
    start_prompt: str = """Consider a question and give an answer - a single integer between 0 and 1000.
Here are a few examples:

<you are asked>
</you are asked>
<you should answer>
</you should answer>


<you are asked>
</you are asked>
<you should answer>
</you should answer>

The question is:

<you are asked>
""",
    end_prompt: str = "\n</you are asked>",
    question_column: str = "input",
    answer_column: str = "output",
    max_samples: int = None,
    min_new_tokens: int = 0,
    max_new_tokens: int = 50,
    remove_suffix: str = None,
    device: str = "cuda",
) -> dict:
    """
    Evaluate a Hugging Face model on a AIME 2024 I task.
    """
    exact_match: list[bool] = []
    model.to(device)  # Ensure the model is on the correct device

    for idx in tqdm(range(min(max_samples, len(data))), desc="Evaluating AIME model"):
        question = data[idx][question_column]
        ground_truth = str(data[idx][answer_column])

        # Generate and decode the output string, removing the special tokens and any suffixes
        decoded = generate_from_prompt(
            model=model,
            tokenizer=tokenizer,
            input_data=question,
            start_prompt=start_prompt,
            end_prompt=end_prompt,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
        )

        # Remove the suffix if specified - note that Mistral-Instruct models add a </s> suffix to specify the end of the output
        if remove_suffix is not None:
            decoded = decoded.replace(remove_suffix, "")

        print(f"Question: {question}")
        print(f"Answer: {decoded}")
        print(f"Ground Truth: {ground_truth}")
        print()

        exact_match.append(compute_exact(decoded, ground_truth))

    return {"exact_match": np.mean(exact_match)}


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Evaluate a model on a AIME 2024 I task.")

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        help="The type of model to evaluate (currently only Huggingface)",
        default="hf",
    )
    parser.add_argument(
        "--hf_model_id",
        type=str,
        help="The Huggingface model to evaluate",
        default="unsloth/llama-3-8b-Instruct-bnb-4bit",
    )

    # Dataset arguments
    parser.add_argument(
        "--max_samples",
        type=int,
        help="The maximum number of samples to evaluate",
        default=200,
    )

    # Generation arguments
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="The maximum number of tokens to generate",
        default=50,
    )
    parser.add_argument(
        "--remove_suffix",
        type=str,
        help="The suffix to remove from the generated output",
        default=None,
    )

    # Environment and reproducibility arguments
    parser.add_argument(
        "--device", type=str, help="The device to use for inference", default="cuda"
    )
    parser.add_argument("--seed", type=int, help="The random seed to use", default=42)
    parser.add_argument(
        "--save_dir",
        type=str,
        help="The directory to save the results to",
        default="results",
    )

    # W&B logging arguments
    parser.add_argument(
        "--wandb_logging", type=str, default="False", help="Whether to log to W&B."
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="aime_2024_i_eval",
        help="The name of the W&B project, for logging.",
    )
    parser.add_argument(
        "--wandb_api_var",
        type=str,
        default="WANDB_API_KEY",
        help="Name of the WandB API key variable name.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)

    # Initialize W&B
    if args.wandb_logging == "True":
        wandb.login(key=getenv(args.wandb_api_var))
        wandb.init(project=args.wandb_name, name=args.run_name, config=args)

    # Load the test split of the dataset
    print("Loading dataset")
    # data = load_dataset(args.dataset, args.dataset_revision, split=args.split)
    data = load_dataset("csv", data_files="data/aime_2024_I.csv", delimiter=";")
    data = data["train"]

    # Model evaluation logic based on the model type
    if args.model_type == "hf":
        # Load the Hugging Face model and tokenizer
        print("Loading Hugging Face model: ", args.hf_model_id)
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id)
        model = AutoModelForCausalLM.from_pretrained(args.hf_model_id).to(args.device)
        model.eval()

        # Evaluate the Hugging Face model
        print("Evaluating Hugging Face model on AIME task: ", args.hf_model_id)
        aime_metrics = evaluate_hf_model_aime(
            model, tokenizer, data, question_column="question", answer_column="answer", max_samples=args.max_samples
        )
    else:
        raise ValueError("Invalid model type: ", args.model_type)

    # Print the metrics to the console
    print("Model AIME Metrics:")
    for key, value in aime_metrics.items():
        print(f"{key}: {value}")

    # Add the model and dataset names to the metrics dictionary
    metrics = {**vars(args), **aime_metrics}

    # Save the metrics to a JSON file
    model_id = args.hf_model_id
    save_path = path.join(
        args.save_dir, f'{model_id.replace("/", "-")}_aime_2024_i_metrics.json'
    )
    print("Saving AIME metrics to: ", save_path)

    if not path.exists(args.save_dir):
        makedirs(args.save_dir)

    with open(save_path, "w") as f:
        json.dump(metrics, f)

    # Log the metrics to W&B
    if args.wandb_logging == "True":
        wandb.log(metrics)
        wandb.finish()
