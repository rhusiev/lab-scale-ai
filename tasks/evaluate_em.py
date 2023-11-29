import evaluate
import numpy as np
import json
import argparse
import torch
import wandb
import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel
from typing import Iterable
from tqdm import tqdm
from os import path, makedirs, getenv

from openai_chat_api import DialogueBot
from generate_from_hf_model import generate_from_prompt


from typing import Optional, List


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

def clean_response(text, positive_value, negative_value):
    gold_toks = get_tokens(text)
    response = gold_toks[0] if len(gold_toks) > 0 else ''
    if response == positive_value:
        return 1
    elif response == negative_value:
        return 0
    else:
        return -1

def calculate_em_metrics(preds: List[int], truths: List[int]):
    """
    Calculate
    """

    # Compute the final metrics
    truths = np.array(truths)
    preds = np.array(preds)

    # Compute the metrics
    N = truths.shape[0]
    accuracy = ((truths == preds)).sum() / N
    TP = ((preds == 1) & (truths == 1)).sum()
    FP = ((preds == 1) & (truths == 0)).sum()
    FN = ((preds == 0) & (truths == 1)).sum()
    TN = ((preds == 0) & (truths == 0)).sum()
    precision = TP / (TP + FP)
    recall = FP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    unreadable_total = ((preds != 0) & (preds != 1)).sum()
    unreadable_positives = ((preds != 0) & (preds != 1) & (truths == 1)).sum()
    unreadable_negatives = ((preds != 0) & (preds != 1) & (truths == 0)).sum()
    unreadable_rate = unreadable_total / N

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'unreadable_rate': unreadable_rate,
        'unreadable_total': unreadable_total,
        'unreadable_positives': unreadable_positives,
        'unreadable_negatives': unreadable_negatives,
    }

############

def evaluate_hf_model_em(model: AutoModelForCausalLM,
                         tokenizer: AutoTokenizer,
                         data: Iterable,
                         input_column: str= 'input',
                         target_column: str= 'output',
                         max_samples: int=None,
                         min_new_tokens: int=0,
                         max_new_tokens: int=50,
                         remove_suffix: str=None,
                         device: str='cuda',
                         start_prompt: str = '### Consider the following question with context: ',
                         end_prompt: str = ' ### Please answer with one of the options listed in the brackets:',
                         positive_value: str = 'y',
                         negative_value: str = 'n'
                         ) -> dict:
    """
    Evaluate a Hugging Face model on a Entity Matching task.
    """
    preds = []
    truths = []
    model.to(device)  # Ensure the model is on the correct device

    for idx in tqdm(range(min(max_samples, len(data))), desc='Evaluating EM model'):
        question = data[idx][input_column]
        ground_truth = data[idx][target_column]

        # Generate and decode the output string, removing the special tokens and any suffixes
        decoded = generate_from_prompt(model=model, 
                                       tokenizer=tokenizer, 
                                       input_data=question, 
                                       start_prompt=start_prompt,
                                       end_prompt=end_prompt, 
                                       min_new_tokens=min_new_tokens,
                                       max_new_tokens=max_new_tokens)

        # Remove the suffix if specified - note that Mistral-Instruct models add a </s> suffix to specify the end of the output
        if remove_suffix is not None:
            decoded = decoded.replace(remove_suffix, '')

        # Since responses may include more tokens than just the requested response token (e.g. 'y' or 'n'),
        # extract the first token from the response
        decoded = clean_response(decoded, positive_value, negative_value)
        ground_truth = clean_response(ground_truth, positive_value, negative_value)

        # Add the decoded and ground truth responses to the list
        preds.append(decoded)
        truths.append(ground_truth)

    metrics = calculate_em_metrics(preds, truths)


    return metrics


def evaluate_openai_model_em(bot: DialogueBot,
                             data: Iterable,
                             input_column: str= 'input',
                             target_column: str= 'output',
                             max_samples: int = None,
                             start_prompt: str = '### Consider the following question with context: ',
                             end_prompt: str = ' ### Please answer with one of the options listed in the brackets:',
                             positive_value: str = 'y',
                             negative_value: str = 'n'
                             ) -> dict:
    """
    Evaluate an OpenAI model on a dataset using EM metrics.

    HAS NOT BEEN TESTED
    """
    preds = []
    truths = []
    # TODO: test and remove this message
    warnings.warn("YOU ARE EVALUATING USING AN EVALUATION FUNCTION THAT HAS NOT BEEN TESTED ")


    # Iterate over the dataset
    for idx in tqdm(range(min(max_samples, len(data))), desc='Evaluating OpenAI EM model'):
        # Create the input string
        input = data[idx][input_column]
        ground_truth = data[idx][target_column]

        # Get the model's response, which is the generated answer
        decoded = bot.return_bot_response(input)

        # Since responses may include more tokens than just the requested response token (e.g. 'y' or 'n'),
        # extract the first token from the response
        decoded = clean_response(decoded, positive_value, negative_value)
        ground_truth = clean_response(ground_truth, positive_value, negative_value)

        # Add the decoded and ground truth responses to the list
        preds.append(decoded)
        truths.append(ground_truth)

    metrics = calculate_em_metrics(preds, truths)


    return metrics


if __name__ == '__main__':
    warnings.warn("YOU ARE RUNNING A SCRIPT THAT HAS NOT BEEN ADAPTED TO EM ")

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a model on an EM task.')

    # Model arguments
    parser.add_argument('--model_type', type=str, help='The type of model to evaluate (Huggingface or OpenAI)', default='hf')
    parser.add_argument('--hf_model_id', type=str, help='The Huggingface model to evaluate', default='llama-2-7b-chat-hf')
    parser.add_argument('--oai_model_id', type=str, help='The OpenAI model ID to use', default='gpt-3.5-turbo')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, help='The dataset to evaluate on', default='cnn_dailymail')
    parser.add_argument('--dataset_revision', type=str, help='The revision of the dataset to use', default='3.0.0')
    parser.add_argument('--split', type=str, help='The split of the dataset to evaluate on', default='test[0:25]')
    parser.add_argument('--input_column', type=str, help='The name of the input column in the dataset', default='article')
    parser.add_argument('--target_column', type=str, help='The name of the target column in the dataset', default='highlights')
    parser.add_argument('--max_samples', type=int, help='The maximum number of samples to evaluate', default=25)

    # Generation arguments
    parser.add_argument('--max_tokens', type=int, help='The maximum number of tokens to generate', default=50)
    parser.add_argument('--remove_suffix', type=str, help='The suffix to remove from the generated output', default=None)

    # Environment and reproducibility arguments
    parser.add_argument('--device', type=str, help='The device to use for inference', default='cuda')
    parser.add_argument('--seed', type=int, help='The random seed to use', default=42)
    parser.add_argument('--save_dir', type=str, help='The directory to save the results to', default='results')

    # W&B logging arguments
    parser.add_argument('--wandb_logging', type=str, default='False', help='Whether to log to W&B.')
    parser.add_argument('--wandb_name', type=str, default='em_eval', help='The name of the W&B project, for logging.')
    parser.add_argument('--wandb_api_var', type=str, default='WANDB_API_KEY', help='Name of the WandB API key variable name.')

    # Parse the arguments
    args = parser.parse_args()

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)

    # Initialize W&B
    if args.wandb_logging == 'True':
        wandb.login(key=getenv(args.wandb_api_var))
        wandb.init(project=args.wandb_name, 
                   name=args.run_name, 
                   config=args)
    
    # Load the test split of the dataset
    print('Loading dataset: ', args.dataset)
    data = load_dataset(args.dataset, args.dataset_revision, split=args.split)

    # Model evaluation logic based on the model type
    if args.model_type == 'hf':
        # Load the Hugging Face model and tokenizer
        print('Loading Hugging Face model: ', args.hf_model_id)
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id)
        model = AutoModelForCausalLM.from_pretrained(args.hf_model_id).to(args.device)
        model.eval()

        # Evaluate the Hugging Face model
        print('Evaluating Hugging Face model on EM task: ', args.hf_model_id)
        em_metrics = evaluate_hf_model_em(model, tokenizer, data, args.input_column, args.target_column, args.max_samples)

    elif args.model_type == 'openai':
        # NOTE: OpenAI Diaglogue bot QandA task has not been tested
        # TODO: Test
        # Evaluate the OpenAI model
        print('Evaluating OpenAI model on QA task: ', args.oai_model_id)
        bot = DialogueBot(model=args.oai_model_id, system_prompt=args.system_prompt)
        qa_metrics = evaluate_hf_model_em(model, tokenizer, data, args.input_column, args.target_column, args.max_samples, args.device)

    else:
        raise ValueError('Invalid model type: ', args.model_type)

    # Print the metrics to the console
    print('Model QA Metrics:')
    for key, value in qa_metrics.items():
        print(f'{key}: {value}')

    # Add the model and dataset names to the metrics dictionary
    metrics = {**vars(args), **qa_metrics}

    # Save the metrics to a JSON file
    model_id = args.hf_model_id if args.model_type == 'hf' else args.oai_model_id
    save_path = path.join(args.save_dir, f'{model_id.replace("/", "-")}_qa_metrics.json')
    print('Saving QA metrics to: ', save_path)

    if not path.exists(args.save_dir):
        makedirs(args.save_dir)

    with open(save_path, 'w') as f:
        json.dump(metrics, f)

    # Log the metrics to W&B
    if args.wandb_logging == 'True':
        wandb.log(metrics)
        wandb.finish()
