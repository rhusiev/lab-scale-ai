#!/usr/bin/env python3

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

from openai_chat_api import DialogueBot
from generate_from_hf_model import generate_from_prompt

def compute_summarization_metrics(predictions: Iterable, 
                            references: Iterable,
                            rouge: bool=True,
                            bleu: bool=True,
                            bertscore: bool=True) -> dict:
    """
    Compute ROUGE, BLEU, and BERTscore metrics for a set of predictions and references.
    """

    metric_results = {}

    if rouge:
        rouge = evaluate.load('rouge')

        # Compute ROUGE metrics at the summary level, using the 'rouge1', 'rouge2', and 'rougeL' metrics, aggregating the results
        rouge_results = rouge.compute(predictions=predictions, 
                                    references=references, 
                                    use_aggregator=True)

        # Store the results in the metric_results dictionary
        metric_results['rouge'] = rouge_results
    
    else:
        metric_results['rouge'] = None

    if bleu:
        bleu = evaluate.load('bleu')

        # Compute BLEU metrics at the summary level
        bleu_results = bleu.compute(predictions=predictions, 
                                    references=references)
        
        # Store the results in the metric_results dictionary
        metric_results['bleu'] = bleu_results
    
    else:
        metric_results['bleu'] = None

    if bertscore:
        bertscore = evaluate.load('bertscore')

        # Compute BERTscore metric, using distilbert-base-uncased as the reference model, and averaging the results
        bertscore_results = bertscore.compute(predictions=predictions, 
                                                    references=references, 
                                                    lang='en', 
                                                    model_type="distilbert-base-uncased")
        
        # Store the results in the metric_results dictionary
        metric_results['bertscore'] = {k: np.mean(v) for k, v in bertscore_results.items() if k in ['precision', 'recall', 'f1']}
    
    else:
        metric_results['bertscore'] = None

    return metric_results

def evaluate_hf_model(model: AutoModelForCausalLM, 
                      tokenizer: AutoTokenizer, 
                      data: Iterable,
                      input_column: str='article',
                      target_column: str='highlights',
                      max_samples: int=None,
                      start_prompt: str='Summarize the following: ',
                      end_prompt: str='\n Begin summary:',
                      max_tokens: int=974,
                      min_new_tokens: int=25,
                      max_new_tokens: int=50,
                      remove_suffix: str=None,
                      rouge: bool=True,
                      bleu: bool=True,
                      bertscore: bool=True) -> dict:
    """
    Evaluate a Hugging Face model on a dataset using three text summarization metrics.
    """
    
    model_outputs = []

    # Iterate over the test set
    for idx in tqdm(range(max_samples), desc='Evaluating Hugging Face model'):

        # Generate and decode the output string, removing the special tokens and any suffixes
        decoded = generate_from_prompt(model, 
                                       tokenizer, 
                                       data[idx][input_column], 
                                       start_prompt, 
                                       end_prompt, 
                                       max_tokens,
                                       min_new_tokens,
                                       max_new_tokens)

        # Remove the suffix if specified - note that Mistral-Instruct models add a </s> suffix to specify the end of the output
        if remove_suffix is not None:
            decoded = decoded.replace(remove_suffix, '')

        model_outputs.append(decoded)
        
    # Compute the ROUGE, BLEU, and BERTscore metrics, comparing the model's responses to the target summaries    
    metrics = compute_summarization_metrics(model_outputs, 
                                            data[target_column][:len(model_outputs)], 
                                            rouge=rouge, 
                                            bleu=bleu, 
                                            bertscore=bertscore)
    
    return model_outputs, metrics

def evaluate_openai_model(bot: DialogueBot,
                          data: Iterable, 
                          input_column: str,
                          target_column: str,
                          max_samples: int=None,
                          start_prompt: str='Summarize the following: ',
                          end_prompt: str='\n Begin summary:',
                          rouge: bool=True,
                          bleu: bool=True,
                          bertscore: bool=True) -> dict:
    """
    Evaluate an OpenAI model on a dataset using three text summarization metrics.
    """

    model_outputs = []
    
    # Iterate over the test set
    for idx in tqdm(range(max_samples), desc='Evaluating OpenAI model'):

        # Create the input string, adding the start and end prompts
        input = start_prompt + data[idx][input_column] + end_prompt
        
        # Get the model's response, omitting the system and user prompts
        output = bot.return_bot_response(input)
        model_outputs.append(output)
    
    # Compute the ROUGE, BLEU, and BERTscore metrics, comparing the model's responses to the target summaries
    metrics = compute_summarization_metrics(model_outputs, 
                                            data[target_column][:len(model_outputs)], 
                                            rouge=rouge, 
                                            bleu=bleu, 
                                            bertscore=bertscore)
    
    return metrics

# Main function
if __name__ == '__main__':

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a text summarization model on a dataset.')

    # Model arguments
    parser.add_argument('--model_type', type=str, help='The type of model to evaluate (Huggingface or OpenAI)', default='hf')
    parser.add_argument('--hf_model_id', type=str, help='The Huggingface model to evaluate', default='facebook/opt-125m')
    parser.add_argument('--oai_model_id', type=str, help='The OpenAI model ID to use in the results file', default='gpt-3.5-turbo')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, help='The dataset to evaluate on', default='cnn_dailymail')
    parser.add_argument('--dataset_revision', type=str, help='The revision of the dataset to use', default='3.0.0')
    parser.add_argument('--split', type=str, help='The split of the dataset to evaluate on', default='test[0:25]')
    parser.add_argument('--input_column', type=str, help='The name of the input column in the dataset', default='article')
    parser.add_argument('--target_column', type=str, help='The name of the target column in the dataset', default='highlights')
    parser.add_argument('--max_samples', type=int, help='The maximum number of samples to evaluate', default=25)

    # Prompt arguments
    parser.add_argument('--system_prompt', type=str, help='The system prompt for the model', default='You are a helpful assistant.')
    parser.add_argument('--start_prompt', type=str, help='The start prompt for the model', default='Summarize the following: ')
    parser.add_argument('--end_prompt', type=str, help='The end prompt for the model', default='\n Begin summary:')
    parser.add_argument('--max_tokens', type=int, help='The maximum number of tokens to generate', default=974)
    parser.add_argument('--remove_suffix', type=str, help='The suffix to remove from the generated output', default=None)

    # Metric arguments
    parser.add_argument('--rouge', type=bool, help='Whether to compute the ROUGE metric', default=True)
    parser.add_argument('--bleu', type=bool, help='Whether to compute the BLEU metric', default=True)
    parser.add_argument('--bertscore', type=bool, help='Whether to compute the BERTscore metric', default=True)

    # PEFT arguments
    parser.add_argument('--peft_model', type=bool, help='Whether to use a PEFT model', default=False)
    parser.add_argument('--peft_dir', type=str, help='The path to the PEFT model config file', default='')
    parser.add_argument('--four_bit', type=bool, help='Whether to use a 4-bit PEFT model', default=False)
    parser.add_argument('--eight_bit', type=bool, help='Whether to use an 8-bit PEFT model', default=False)

    # Generation arguments
    parser.add_argument('--min_new_tokens', type=int, help='The minimum number of new tokens to generate', default=25)
    parser.add_argument('--max_new_tokens', type=int, help='The maximum number of new tokens to generate', default=50)

    # Environment and reproducibility arguments
    parser.add_argument('--device', type=str, help='The device to use for inference', default='cpu')
    parser.add_argument('--seed', type=int, help='The random seed to use', default=42)
    parser.add_argument('--save_dir', type=str, help='The directory to save the results to', default='results')

    # W&B logging arguments
    parser.add_argument('--wandb_logging', type=str, default='False', help='Whether to log to W&B.')
    parser.add_argument('--wandb_name', type=str, default='summarization_eval', help='The name of the W&B project, for logging.')
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

    if args.dataset_revision:
        data = load_dataset(args.dataset, args.dataset_revision, split=args.split)

    else:
        data = load_dataset(args.dataset, split=args.split)

    # HF model
    if args.model_type == 'hf':

        # Load the Hugging Face model and tokenizer
        print('Loading Hugging Face model: ', args.hf_model_id)

        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # If the model is a PEFT model, load the PEFT model and tokenizer
        if args.peft_model:

            # Load the PEFT model in the specified precision
            if args.four_bit:                
                model = AutoModelForCausalLM.from_pretrained(args.hf_model_id, load_in_4bit=True).to(args.device)
            elif args.eight_bit:
                model = AutoModelForCausalLM.from_pretrained(args.hf_model_id, load_in_8bit=True).to(args.device)
            
            # Get the PEFT model
            model = PeftModel.from_pretrained(model, args.peft_dir).to(args.device)

        # If the model is not a PEFT model, load the Hugging Face model and tokenizer
        else:
            model = AutoModelForCausalLM.from_pretrained(args.hf_model_id).to(args.device)

        # Set the model to evaluation mode
        model.eval()

        # Evaluate the Hugging Face model
        print('Evaluating Hugging Face model: ', args.hf_model_id)

        metrics = evaluate_hf_model(model, 
                                    tokenizer, 
                                    data, 
                                    args.input_column, 
                                    args.target_column, 
                                    args.max_samples, 
                                    args.start_prompt, 
                                    args.end_prompt, 
                                    args.max_tokens,
                                    args.min_new_tokens,
                                    args.max_new_tokens,
                                    args.remove_suffix, 
                                    args.rouge, 
                                    args.bleu, 
                                    args.bertscore)

    # OpenAI model
    elif args.model_type == 'openai':

        # Evaluate the OpenAI model
        print('Evaluating OpenAI model: ', args.oai_model_id)

        bot = DialogueBot(model=args.oai_model_id, system_prompt=args.system_prompt)
        metrics = evaluate_openai_model(bot, 
                                        data, 
                                        args.input_column, 
                                        args.target_column, 
                                        args.max_samples, 
                                        args.start_prompt, 
                                        args.end_prompt, 
                                        args.rouge, 
                                        args.bleu, 
                                        args.bertscore)

    else:
        raise ValueError('Invalid model type: ', args.model_type)

    # Print the metrics to the console
    print('Model Summarization Metrics')

    for key, value in metrics.items():
        print(f'{key}: {value}')
    
    # Add the model and dataset names to the metrics dictionary
    for key, value in vars(args).items():

        # Don't overwrite ROUGE, BLEU, or BERTscore metrics
        if key not in metrics:

            # Add vars to the metrics dictionary
            metrics[key] = value

    # Get the model ID for saving from the command line arguments
    model_id = args.hf_model_id if args.model_type == 'hf' else args.oai_model_id

    # Save the metrics to a JSON file
    print('Saving metrics to: ', f'{args.save_dir}/{model_id.replace("/", "-")}_metrics.json')

    if not path.exists(args.save_dir):
        makedirs(args.save_dir)

    with open(path.join(args.save_dir, f'{model_id.replace("/", "-")}_metrics.json'), 'w') as f:
        json.dump(metrics, f)

    # Log the metrics to W&B
    if args.wandb_logging == 'True':
        wandb.log(metrics)
        wandb.finish()
