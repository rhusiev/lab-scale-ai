#!/usr/bin/env python3

import evaluate
import json
import argparse
import torch
import wandb
import pickle as pkl
import os

import numpy as np
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel
from typing import Iterable
from tqdm import tqdm
from os import path, makedirs, getenv

from openai_chat_api import DialogueBot
from generate_from_hf_model import generate_from_prompt
from finetune import QUANZATION_MAP

# Default instructions and k-shot prompts for the fact-checking task and the Climate-FEVER dataset
FACT_CHECK_INSTRUCTION = 'Please classify the following Claim into one of three labels based on the Evidence that follows it. The labels are SUPPORTS, if the Claim is supported by the Evidence; REFUTES, if the Claim is refuted by the Evidence; or NOT_ENOUGH_INFO, if the Claim is neither supported nor refuted by the Evidence. Output only one word: SUPPORTS, REFUTES, or NOT_ENOUGH_INFO. '

FACT_CHECK_ONE_SHOT = '\nHere is an example of how to perform the task. Claim: The Review concluded that CRU\'s actions were normal and did not threaten the integrity of peer review. Evidence: Describing its report as "hugely positive", he stated that "it is especially important that, despite a deluge of allegations and smears against the CRU, this independent group of utterly reputable scientists have concluded that there was no evidence of any scientific malpractice." What is the correct label? SUPPORTS\n'
FACT_CHECK_TWO_SHOT = 'Here is another example of how to perform the task. Claim: Recent research also indicates that the quantity of fossil fuels staying in the atmosphere is much less than previously thought. Evidence: (2007) concluded that unless energy policies changed substantially, the world would continue to depend on fossil fuels until 2025–2030. What is the correct label? NOT_ENOUGH_INFO\n'
FACT_CHECK_THREE_SHOT = 'Here is another example of how to perform the task. Claim: "Global warming" and "climate change" mean different things and have both been used for decades. Evidence: Global warming and climate change are often used interchangeably. What is the correct label? REFUTES\n'

FACT_CHECK_TRANSITION = 'Here is the example that needs to be classified. Please respond with only one word after being asked for the correct label. '

# Default chat tokens, end prompts, and suffixes for each model
MODEL_CHAT_TOKENS = {
    'openai': '',
    'mistral': '<s>[INST] ',
    'llama-2': '<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n',
    'falcon': 'A helpful assistant.\nUser: ',
    'opt-finetune': '',
}

MODEL_END_PROMPTS = {
    'openai': ' What is the correct label?',
    'mistral': ' What is the correct label? [/INST]',
    'llama-2': ' What is the correct label? [/INST]',
    'falcon': ' What is the correct label?\nAssistant:',
    'opt-finetune': ' What is the correct label?',
}

MODEL_SUFFIXES = {
    'openai': '',
    'mistral': '</s>',
    'llama-2': '</s>',
    'falcon': '<|endoftext|>',
    'opt-finetune': '</s>',
}

def compute_classification_metrics(predictions: Iterable,
                                   references: Iterable,
                                   convert_to_int: bool=True,
                                   vals: tuple=('SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO')) -> dict:
    """
    Compute classification metrics for binary or categorical data - accuracy, F1, precision, recall.
    """

    print(predictions)
    print(references)

    if convert_to_int:

        if not vals:
            vals = list(set(predictions))

        max_ = len(vals)

        val_map = {vals[i]:i for i in range(len(vals))}
        predictions = [max_ if predictions[i] not in val_map else val_map[predictions[i]] for i in range(len(predictions))]
        references = [max_ if references[i] not in val_map else val_map[references[i]] for i in range(len(references))]

    accuracy = evaluate.load('accuracy')
    f1 = evaluate.load('f1')
    precision = evaluate.load('precision')
    recall = evaluate.load('recall')

    classification_metrics = {'f1': f1, 'precision': precision, 'recall': recall}
    classification_metrics = {key: value.compute(predictions=predictions, references=references, average='weighted') for key, value in classification_metrics.items()}
    classification_metrics['accuracy'] = accuracy.compute(predictions=predictions, references=references)

    print(predictions)
    print(references)

    return classification_metrics

def evaluate_hf_classifications(model: AutoModelForCausalLM, 
                      tokenizer: AutoTokenizer, 
                      data: Iterable,
                      input_column: str='concat_col',
                      target_column: str='evidence_label',
                      max_samples: int=None,
                      start_prompt: str='Classify as supported, refuted, or not enough evidence: ',
                      end_prompt: str='\n Classification:',
                      max_tokens: int=974,
                      min_new_tokens: int=25,
                      max_new_tokens: int=50,
                      remove_suffix: str=None,
                      save_output_dir: str=None,
                      run_name: str='',
                      remove_stop_tokens: Iterable=None) -> dict:
    """
    Evaluate a Hugging Face model on a dataset using three classification metrics.
    """

    model_outputs = []

    if not max_samples:
        max_samples = len(data)

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

        # Remove the suffix if specified
        if remove_suffix is not None and remove_suffix in decoded:
            decoded = decoded.split(remove_suffix)[0]

        # Remove the stop tokens if specified
        if remove_stop_tokens is not None:
            for token in remove_stop_tokens:
                decoded = decoded.replace(token, '')

        model_outputs.append(decoded.strip())
        
    # Compute the classification metrics, comparing the model's responses to the target labels    
    metrics = compute_classification_metrics(model_outputs, 
                                            data[target_column][:len(model_outputs)])

    # Save model outputs to dataframe
    if save_output_dir:
        output_df = pd.DataFrame(np.column_stack([data[input_column][:len(model_outputs)], data[target_column][:len(model_outputs)], model_outputs]),
                                 columns=['input', 'label', 'output'])
        output_df.to_csv(path.join(save_output_dir, f'{run_name}_outputs.csv'))

    return metrics

def evaluate_openai_classifications(bot: DialogueBot,
                          data: Iterable, 
                          input_column: str='concat_col',
                          target_column: str='evidence_label',
                          max_samples: int=None,
                          start_prompt: str='Summarize the following: ',
                          end_prompt: str='\n Begin summary:',
                          save_output_dir: str=None,
                          run_name: str='',
                          remove_stop_tokens: Iterable=None,
                          intermediate_outputs_dir: str=None) -> dict:
    """
    Evaluate an OpenAI model on a dataset using three classification metrics.
    """

    model_outputs = []
    intermediate_idx = 0

    # Load the intermediate outputs - all pickles
    files = [i for i in os.listdir(intermediate_outputs_dir) if i.endswith('.pkl')]

    if files:

        with open(os.path.join(intermediate_outputs_dir, files[0]), 'rb') as f:
            model_outputs.extend(pkl.load(f))

    # If no intermediate outputs, start from the beginning; otherwise, start from the last index
    start_idx = len(model_outputs)

    if not max_samples:
        max_samples = len(data)

    # Create progress bar
    pbar = tqdm(total=max_samples, desc='Evaluating OpenAI model')
    pbar.update(start_idx)
    
    # Iterate over the test set
    for idx in range(start_idx, max_samples):

        # Create the input string, adding the start and end prompts
        input = start_prompt + data[idx][input_column] + end_prompt
        
        # Get the model's response, omitting the system and user prompts
        try:
            output = bot.return_bot_response(input)
        except:
            pkl.dump(model_outputs, open(os.path.join(intermediate_outputs_dir, f'intermediate_outputs.pkl'), 'wb'))
            raise ValueError('OpenAI API error')

        # Remove the stop tokens if specified
        if remove_stop_tokens is not None:
            for token in remove_stop_tokens:
                output = output.replace(token, '')

        model_outputs.append(output.strip())
    
        # Update the progress bar
        pbar.update(1)

    # Compute the classification metrics, comparing the model's responses to the target labels    
    metrics = compute_classification_metrics(model_outputs, 
                                            data[target_column][:len(model_outputs)])
    
    # Save model outputs to dataframe
    if save_output_dir:
        output_df = pd.DataFrame(np.column_stack([data[input_column][:len(model_outputs)], data[target_column][:len(model_outputs)], model_outputs]),
                                 columns=['input', 'label', 'output'])
        output_df.to_csv(path.join(save_output_dir, f'{run_name}_outputs.csv'))

    return metrics

# Main function
if __name__ == '__main__':

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a model on a dataset requiring classification.')

    # Model arguments
    parser.add_argument('--model_type', type=str, help='The type of model to evaluate (Huggingface or OpenAI)', default='hf')
    parser.add_argument('--hf_model_id', type=str, help='The Huggingface model to evaluate', default='mistralai/Mistral-7B-Instruct-v0.1')
    parser.add_argument('--oai_model_id', type=str, help='The OpenAI model ID to use in the results file', default='gpt-3.5-turbo')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, help='The dataset to evaluate on', default='amandakonet/climate_fever_adopted')
    parser.add_argument('--dataset_revision', type=str, help='The revision of the dataset to use', default='')
    parser.add_argument('--split', type=str, help='The split of the dataset to evaluate on', default='test')
    parser.add_argument('--input_column', type=str, help='The name of the input column in the dataset', default='concat_col')
    parser.add_argument('--target_column', type=str, help='The name of the target column in the dataset', default='evidence_label')
    parser.add_argument('--max_samples', type=int, help='The maximum number of samples to evaluate', default=None)

    # Handle case where claim and evidence is separated
    parser.add_argument('--separate_inputs', type=bool, help='Specify if multiple input columns should be concatenated', default=True)
    parser.add_argument('--firstinput', type=str, help='Name of the first input column', default='claim')
    parser.add_argument('--firstinputflag', type=str, help='Name of the first input column', default='Claim: ')
    parser.add_argument('--secondinput', type=str, help='Name of the first input column', default='evidence')
    parser.add_argument('--secondinputflag', type=str, help='Name of the first input column', default='Evidence: ')

    # Prompt arguments
    parser.add_argument('--use_model_prompt_defaults', type=str, help='Whether to use the default prompts for a model', default='mistral')
    parser.add_argument('--system_prompt', type=str, help='The system prompt for the model', default='You are a helpful assistant.')
    parser.add_argument('--start_prompt', type=str, help='The start prompt for the model', default=FACT_CHECK_INSTRUCTION)
    parser.add_argument('--end_prompt', type=str, help='The end prompt for the model', default=' Label?')
    parser.add_argument('--max_tokens', type=int, help='The maximum number of tokens to generate', default=974)
    parser.add_argument('--remove_suffix', type=str, help='The suffix to remove from the generated output', default=None)
    parser.add_argument('--remove_stop_tokens', type=str, help='Stop tokens to remove from generated output separated by +', default='.')

    # Few-shot arguments
    parser.add_argument('--shots', type=int, help='The number of shots to use for the model', default=0)
    parser.add_argument('--first_shot', type=str, help='The first shot to use for the model', default=FACT_CHECK_ONE_SHOT)
    parser.add_argument('--second_shot', type=str, help='The second shot to use for the model', default=FACT_CHECK_TWO_SHOT)
    parser.add_argument('--third_shot', type=str, help='The third shot to use for the model', default=FACT_CHECK_THREE_SHOT)
    parser.add_argument('--transition', type=str, help='The transition to use between shots', default=FACT_CHECK_TRANSITION)

    # PEFT arguments
    parser.add_argument('--peft_model', type=bool, help='Whether to use a PEFT model', default=False)
    parser.add_argument('--peft_dir', type=str, help='The path to the PEFT model config file', default='')
    parser.add_argument('--four_bit', type=bool, help='Whether to use a 4-bit PEFT model', default=True)
    parser.add_argument('--eight_bit', type=bool, help='Whether to use an 8-bit PEFT model', default=False)

    # Generation arguments
    parser.add_argument('--min_new_tokens', type=int, help='The minimum number of new tokens to generate', default=1)
    parser.add_argument('--max_new_tokens', type=int, help='The maximum number of new tokens to generate', default=10)

    # Environment and reproducibility arguments
    parser.add_argument('--device', type=str, help='The device to use for inference', default='cuda:0')
    parser.add_argument('--seed', type=int, help='The random seed to use', default=42)
    parser.add_argument('--results_dir', type=str, help='The directory to save the results to', default='results')
    parser.add_argument('--intermediate_outputs_dir', type=str, help='The directory to save the intermediate outputs to', default='intermediate_outputs')
    parser.add_argument('--run_name', type=str, default='fact_checking_eval', help='The name of the project, for logging.')

    # W&B logging arguments
    parser.add_argument('--wandb_logging', type=str, default='True', help='Whether to log to W&B.')
    parser.add_argument('--wandb_name', type=str, default='fact_checking_eval', help='The name of the W&B project, for logging.')
    parser.add_argument('--wandb_api_var', type=str, default='WANDB_API_KEY', help='Name of the WandB API key variable name.')

    # Parse the arguments
    args = parser.parse_args()

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)

    # Update the run name
    args.run_name = f'{args.run_name}_{args.shots}-shot'

    # Initialize W&B
    if args.wandb_logging == 'True':
        wandb.login(key=getenv(args.wandb_api_var))
        wandb.init(project=args.wandb_name, 
                   name=args.run_name, 
                   config=args)
    
    # Create results directory
    if not path.exists(args.results_dir):
        makedirs(args.results_dir)

    # Update the start and end prompts if using the model defaults
    if args.use_model_prompt_defaults:

        args.start_prompt = MODEL_CHAT_TOKENS[args.use_model_prompt_defaults] + args.start_prompt
        args.end_prompt = MODEL_END_PROMPTS[args.use_model_prompt_defaults]
        args.remove_suffix = MODEL_SUFFIXES[args.use_model_prompt_defaults]
    
    # Add shots to the start prompt if specified
    if args.shots > 0:

        if args.shots == 1:
            args.start_prompt = args.start_prompt + args.first_shot + args.transition
        elif args.shots == 2:
            args.start_prompt = args.start_prompt + args.first_shot + args.second_shot + args.transition
        elif args.shots == 3:
            args.start_prompt = args.start_prompt + args.first_shot + args.second_shot + args.third_shot + args.transition
        else:
            raise ValueError('Invalid number of shots: ', args.shots)

    # Create list of stop tokens to remove
    if args.remove_stop_tokens:
        args.remove_stop_tokens = args.remove_stop_tokens.split('+')
    
    # Load the test split of the dataset
    print('Loading dataset: ', args.dataset)

    if args.dataset_revision:
        data = load_dataset(args.dataset, args.dataset_revision, split=args.split)

    else:
        data = load_dataset(args.dataset, split=args.split)

    # If input uses multiple columns, concatenate into a new column
    if args.separate_inputs:

        concat_col = [f'{args.firstinputflag}{data[args.firstinput][i]} {args.secondinputflag}{data[args.secondinput][i]}' for i in range(len(data))]
        data = data.add_column('concat_col', concat_col)
        args.input_column = 'concat_col'

    # HF model
    if args.model_type == 'hf':

        # Load the Hugging Face model and tokenizer
        print('Loading Hugging Face model: ', args.hf_model_id)

        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load the quantized model in the specified precision
        if args.four_bit:                
            model = AutoModelForCausalLM.from_pretrained(args.hf_model_id, quantization_config=QUANZATION_MAP['4bit'])
        
        elif args.eight_bit:
            model = AutoModelForCausalLM.from_pretrained(args.hf_model_id, quantization_config=QUANZATION_MAP['8bit'])

        # If the model is not a quantized model, load the Hugging Face model and tokenizer
        else:
            model = AutoModelForCausalLM.from_pretrained(args.hf_model_id).to(args.device)
        
        # If the model is a PEFT model, load the PEFT model and tokenizer
        if args.peft_model:

            # Get the PEFT model
            model = PeftModel.from_pretrained(model, args.peft_dir)

        # Set the model to evaluation mode
        model.eval()

        # Evaluate the Hugging Face model
        print('Evaluating Hugging Face model: ', args.hf_model_id)

        metrics = evaluate_hf_classifications(model, 
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
                                    args.results_dir,
                                    args.run_name,
                                    args.remove_stop_tokens)

    # OpenAI model
    elif args.model_type == 'openai':

        args.intermediate_outputs_dir = f'{args.intermediate_outputs_dir}_{args.shots}-shot'

        # Create intermediate outputs directory
        if not path.exists(args.intermediate_outputs_dir):
            makedirs(args.intermediate_outputs_dir)

        # Evaluate the OpenAI model
        print('Evaluating OpenAI model: ', args.oai_model_id)

        bot = DialogueBot(model=args.oai_model_id, system_prompt=args.system_prompt)
        metrics = evaluate_openai_classifications(bot, 
                                        data, 
                                        args.input_column, 
                                        args.target_column, 
                                        args.max_samples, 
                                        args.start_prompt, 
                                        args.end_prompt,
                                        args.results_dir,
                                        args.run_name,
                                        args.remove_stop_tokens,
                                        args.intermediate_outputs_dir)

    else:
        raise ValueError('Invalid model type: ', args.model_type)

    # Log the metrics to W&B
    if args.wandb_logging == 'True':
        wandb.log(metrics)
        wandb.finish()

    # Print the metrics to the console
    print('Model Classification Metrics')

    for key, value in metrics.items():
        print(f'{key}: {value}')
    
    # Add the model and dataset names to the metrics dictionary
    for key, value in vars(args).items():

        # Don't overwrite classification metrics
        if key not in metrics:

            # Add vars to the metrics dictionary
            metrics[key] = value

    # Get the model ID for saving from the command line arguments
    model_id = args.hf_model_id if args.model_type == 'hf' else args.oai_model_id

    # Save the metrics to a JSON file
    print('Saving metrics to: ', f'{args.results_dir}/{args.run_name}_metrics.json')

    with open(path.join(args.results_dir, f'{args.run_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f)
