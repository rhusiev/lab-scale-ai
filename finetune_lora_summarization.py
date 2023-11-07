#!/usr/bin/env python3

import torch
import bitsandbytes as bnb
import logging
import sys
import transformers
import datasets
import argparse
import wandb

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling, AutoModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import login as hf_login
from tqdm import tqdm
from os import path, mkdir, getenv
from typing import Mapping, Iterable

from evaluate_summarization import evaluate_hf_model

QUANZATION_MAP = {
    '4bit': BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    '8bit': BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["lm_head"],
        torch_dtype=torch.bfloat16,
    ),
}

DEFAULT_TRAINING_ARGS = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=50,
        learning_rate=2e-4,
        fp16=True if torch.cuda.is_available() else False,
        logging_steps=1,
        output_dir='outputs',
        optim='paged_adamw_8bit' if torch.cuda.is_available() else 'adamw_torch',
        use_mps_device=False,
        log_level='info',
        logging_first_step=True,
        evaluation_strategy='steps',
        eval_steps=25
    )

def format_data_as_instructions(data: Mapping, 
                                input_field: str='article', 
                                target_field: str='highlights', 
                                start_prompt: str=' ### Summarize the following: ', 
                                end_prompt: str=' ### Begin summary: ', 
                                suffix: str='') -> list[str]:
    """
    Formats text data as instructions for the model. Can be used as a formatting function for the trainer class.
    """

    output_texts = []

    # Iterate over the data and format the text
    for i in tqdm(range(len(data[input_field])), desc='Formatting data'):

        # Add the start and end prompts to the text, and append the suffix if provided
        text = f'{start_prompt}{data[input_field][i]}{end_prompt}{data[target_field][i]}{suffix}'

        output_texts.append(text)

    return output_texts

def get_model_and_tokenizer(model_id: str, 
                            quantization_type: str='', 
                            gradient_checkpointing: bool=True, 
                            device: str='cpu') -> tuple[AutoModel, AutoTokenizer]:
    """
    Returns a Transformers model and tokenizer for fine-tuning. If quantization_type is provided, the model will be quantized and prepared for training.
    """

    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Set the pad token (needed for trainer class, no value by default for most causal models)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Download the model, quantize if requested
    if quantization_type:
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=QUANZATION_MAP[quantization_type], device_map=device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)

    # Enable gradient checkpointing if requested
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Prepare the model for training if quantization is requested
    if quantization_type:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer

def find_lora_modules(model: AutoModel, 
                      include_modules: Iterable=(bnb.nn.Linear4bit), 
                      exclude_names: Iterable=('lm_head')) -> list[str]:
    """
    Returns a list of the modules to be tuned using LoRA.
    """

    # Create a set to store the names of the modules to be tuned
    lora_module_names = set()

    # Iterate over the model and find the modules to be tuned
    for name, module in model.named_modules():

        # Check if the module is in the list of modules to be tuned
        if any(isinstance(module, include_module) for include_module in include_modules):

            # Split the name of the module and add it to the set
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # Return the list of module names to be tuned, excluding any names in the exclude list
    return [name for name in list(lora_module_names) if name not in exclude_names]

def get_lora_model(model: AutoModel,
                   matrix_rank: int=8,
                   scaling_factor: int=32,
                   dropout: float=0.05,
                   bias: str='none',
                   task_type: str='CAUSAL_LM',
                   include_modules: Iterable=(bnb.nn.Linear4bit),
                   exclude_names: Iterable=('lm_head')) -> AutoModel:
    """
    Returns a model with LoRA applied to the specified modules.
    """

    config = LoraConfig(
        r=matrix_rank,
        lora_alpha=scaling_factor,
        target_modules=find_lora_modules(model, include_modules, exclude_names),
        lora_dropout=dropout,
        bias=bias,
        task_type=task_type,
    )

    return get_peft_model(model, config)

def get_summarization_dataset(dataset: str,
                              streaming: bool=False,
                              split: str='', 
                              instruction_format: bool=False,
                              input_field: str='article',
                              target_field: str='highlights',
                              start_prompt: str=' ### Summarize the following: ',
                              end_prompt: str=' ### Begin summary: ',
                              suffix: str='',
                              pretokenize: bool=False, 
                              tokenizer: AutoTokenizer=None,
                              max_tokens: int=974) -> dict:
    """
    Returns a dataset for summarization fine-tuning, formatted and tokenized as specified.
    """

    # Download the dataset
    data = load_dataset(dataset, streaming=streaming, split=split)

    # Format the data as instructions if requested
    if instruction_format:
        data = format_data_as_instructions(data, input_field, target_field, start_prompt, end_prompt, suffix)

    # Pretokenize the data if requested
    if pretokenize:
        data = data.map(lambda x: tokenizer(x, truncation=True, max_length=max_tokens), batched=True)

    # Return the dataset
    return data

def get_dataset_slices(dataset: str,
                       version: str='',
                       train_slice: str='train[:1000]',
                       validation_slice: str='validation[:25]',
                       test_slice: str='test[:25]') -> dict:
    """
    Returns a dictionary of subsets of the training, validation, and test splits of a dataset.
    """

    # Download the dataset splits, including the dataset version if specified
    if version:
        train_data = load_dataset(dataset, version=version, split=train_slice)
        validation_data = load_dataset(dataset, version=version, split=validation_slice)
        test_data = load_dataset(dataset, version=version, split=test_slice)
    else:
        train_data = load_dataset(dataset, split=train_slice)
        validation_data = load_dataset(dataset, split=validation_slice)
        test_data = load_dataset(dataset, split=test_slice)

    # Return the dictionary of dataset splits
    return {'train': train_data, 'validation': validation_data, 'test': test_data}
    
def get_default_trainer(model: AutoModel,
                tokenizer: AutoTokenizer,
                train_dataset: Mapping,
                eval_dataset: Mapping=None,
                formatting_func: callable=format_data_as_instructions,                
                max_seq_length: int=974,
                training_args: TrainingArguments=None) -> SFTTrainer:
    """
    Returns the default trainer for fine-tuning a summarization model based on the specified training config.
    """

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args if training_args else DEFAULT_TRAINING_ARGS,
        formatting_func=formatting_func,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        max_seq_length=max_seq_length,
        packing=False,
    )

    return trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune a summarization model.')

    # Model ID
    parser.add_argument('--model_id', type=str, default='facebook/opt-125m', help='The model ID to fine-tune.')
    parser.add_argument('--hf_token_var', type=str, default='HF_TOKEN', help='Name of the HuggingFace API token variable name.')

    # Device arguments
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to mount the model on.')
    parser.add_argument('--use_mps_device', type=str, default='False', help='Whether to use an MPS device.')

    # Model arguments
    parser.add_argument('--gradient_checkpointing', type=str, default='True', help='Whether to use gradient checkpointing.')
    parser.add_argument('--quantization_type', type=str, default='4bit', help='The quantization type to use for fine-tuning.')
    parser.add_argument('--lora', type=str, default='True', help='Whether to use LoRA.')
    parser.add_argument('--tune_modules', type=str, default='linear4bit', help='The modules to tune using LoRA.')
    parser.add_argument('--exclude_names', type=str, default='lm_head', help='The names of the modules to exclude from tuning.')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cnn_dailymail', help='The dataset to use for fine-tuning.')
    parser.add_argument('--version', type=str, default='3.0.0', nargs='?', help='The version of the dataset to use for fine-tuning.')
    parser.add_argument('--input_col', type=str, default='article', help='The name of the input column in the dataset.')
    parser.add_argument('--target_col', type=str, default='highlights', help='The name of the target column in the dataset.')
    parser.add_argument('--train_slice', type=str, default='train[:50]', help='The slice of the training dataset to use for fine-tuning.')
    parser.add_argument('--validation_slice', type=str, default='validation[:10]', help='The slice of the validation dataset to use for fine-tuning.')
    parser.add_argument('--test_slice', type=str, default='test[:10]', help='The slice of the test dataset to use for fine-tuning.')

    # Saving arguments
    parser.add_argument('--save_model', type=str, default='True', help='Whether to save the fine-tuned model and tokenizer.')
    parser.add_argument('--save_dir', type=str, default='finetuned_model', help='The directory to save the fine-tuned model and tokenizer.')
    parser.add_argument('--peft_save_dir', type=str, default='peft_model', help='The directory to save the PEFT model.')

    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='logs', help='The directory to save the log file.')
    parser.add_argument('--log_level', type=str, default='info', help='The log level to use for fine-tuning.')
    parser.add_argument('--logging_first_step', type=str, default='True', help='Whether to log the first step.')
    parser.add_argument('--logging_steps', type=int, default=1, help='The number of steps between logging.')
    parser.add_argument('--run_name', type=str, default='peft_finetune', help='The name of the run, for logging.')

    # W&B logging arguments
    parser.add_argument('--wandb_logging', type=str, default='True', help='Whether to log to W&B.')
    parser.add_argument('--wandb_name', type=str, default='peft_finetune', help='The name of the W&B project, for logging.')
    parser.add_argument('--wandb_api_var', type=str, default='WANDB_API_KEY', help='Name of the WandB API key variable name.')

    # Prompt arguments
    parser.add_argument('--start_prompt', type=str, default=' ### Summarize the following: ', help='The start prompt to add to the beginning of the input text.')
    parser.add_argument('--end_prompt', type=str, default=' ### Begin summary: ', help='The end prompt to add to the end of the input text.')
    parser.add_argument('--suffix', type=str, default='', help='The suffix to add to the end of the input and target text.')
    parser.add_argument('--max_seq_length', type=int, default=974, help='The maximum sequence length to use for fine-tuning.')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use for fine-tuning.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='The number of gradient accumulation steps to use for fine-tuning.')
    parser.add_argument('--warmup_steps', type=int, default=10, help='The number of warmup steps to use for fine-tuning.')
    parser.add_argument('--max_steps', type=int, default=50, help='The maximum number of steps to use for fine-tuning.')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='The learning rate to use for fine-tuning.')
    parser.add_argument('--fp16', type=str, default='True', help='Whether to use fp16.')
    parser.add_argument('--output_dir', type=str, default='outputs', help='The directory to save the fine-tuned model.')
    parser.add_argument('--optim', type=str, default='paged_adamw_8bit', help='The optimizer to use for fine-tuning.')

    # Evaluation arguments
    parser.add_argument('--evaluation_strategy', type=str, default='steps', help='The evaluation strategy to use for fine-tuning.')
    parser.add_argument('--eval_steps', type=int, default=25, help='The number of steps between evaluations.')
    parser.add_argument('--eval_on_test', type=str, default='True', help='Whether to evaluate the model on the test set after fine-tuning.')
    parser.add_argument('--compute_summarization_metrics', type=str, default='True', help='Whether to evaluate the model on ROUGE, BLEU, and BERTScore after fine-tuning.')
    
    # Hub arguments
    parser.add_argument('--hub_upload', type=str, default='False', help='Whether to upload the model to the hub.')
    parser.add_argument('--hub_save_id', type=str, default='wolferobert3/opt-125m-peft-summarization', help='The name under which the mode will be saved on the hub.')

    # Parse arguments
    args = parser.parse_args()

    # Define a data formatter function that wraps the format_data_as_instructions function with the specified arguments
    def data_formatter(data: Mapping,
                       input_field: str=args.input_col,
                       target_field: str=args.target_col,
                       start_prompt: str=args.start_prompt,
                       end_prompt: str=args.end_prompt,
                       suffix: str=args.suffix) -> list[str]:
        """
        Wraps the format_data_as_instructions function with the specified arguments.
        """

        return format_data_as_instructions(data, input_field, target_field, start_prompt, end_prompt, suffix)

    # HF Login
    if args.hf_token_var:
        hf_login(getenv(args.hf_token_var))

    # Initialize W&B
    if args.wandb_logging == 'True':
        wandb.login(key=getenv(args.wandb_api_var))
        wandb.init(project=args.wandb_name, 
                   name=args.run_name, 
                   config=args)
    
    # Create directories if they do not exist
    if not path.exists(args.peft_save_dir):
        mkdir(args.peft_save_dir)
        print(f'Created directory {args.peft_save_dir}')
    
    if not path.exists(args.log_dir):
        mkdir(args.log_dir)
        print(f'Created directory {args.log_dir}')

    # Create a logger
    logger = logging.getLogger(__name__)

    # Setup logging
    print('Setting up logging...')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Use the default log level matching the training args
    log_level = DEFAULT_TRAINING_ARGS.get_process_log_level()
    logger.setLevel(log_level)

    # Set the log level for the transformers and datasets libraries
    transformers.utils.logging.get_logger("transformers").setLevel(log_level)
    datasets.utils.logging.get_logger("datasets").setLevel(log_level)

    # Log to file
    file_handler = logging.FileHandler(path.join(args.log_dir, f'{args.run_name}.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(file_handler)

    # Get model and tokenizer
    print('Getting model and tokenizer...')

    model, tokenizer = get_model_and_tokenizer(args.model_id,
                                               quantization_type=args.quantization_type,
                                               gradient_checkpointing=bool(args.gradient_checkpointing),
                                               device=args.device)

    logger.info(f'Loaded Model ID: {args.model_id}')

    # Get LoRA model
    if args.lora == 'True':

        print('Getting LoRA model...')

        if args.tune_modules == 'linear':
            lora_modules = [torch.nn.Linear]
        elif args.tune_modules == 'linear4bit':
            lora_modules = [bnb.nn.Linear4bit]
        elif args.tune_modules == 'linear8bit':
            lora_modules = [bnb.nn.Linear8bit]
        else:
            raise ValueError(f'Invalid tune_modules argument: {args.tune_modules}, must be linear, linear4bit, or linear8bit')

        model = get_lora_model(model,
                               include_modules=lora_modules,
                               exclude_names=args.exclude_names)

        logger.info(f'Loaded LoRA Model')
    
    # Download and prepare data
    print('Downloading and preparing data...')

    data = get_dataset_slices(args.dataset,
                              args.version,
                              train_slice=args.train_slice,
                              validation_slice=args.validation_slice,
                              test_slice=args.test_slice)

    # Set the format of the data
    train_data = data['train']
    train_data.set_format(type='torch', device=args.device)

    validation_data = data['validation']
    validation_data.set_format(type='torch', device=args.device)

    logger.info(f'Loaded Dataset: {args.dataset}')

    # Instantiate trainer
    print('Instantiating trainer...')

    training_args = TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=args.fp16 == 'True',
            logging_steps=args.logging_steps,
            output_dir='outputs',
            optim=args.optim,
            use_mps_device=args.use_mps_device == 'True',
            log_level=args.log_level,
            logging_first_step=args.logging_first_step == 'True',
            evaluation_strategy=args.evaluation_strategy,
            eval_steps=args.eval_steps,
            report_to=['wandb'] if args.wandb_logging == 'True' else [],
        )

    trainer = get_default_trainer(model, 
                                  tokenizer, 
                                  data['train'], 
                                  eval_dataset=data['validation'],
                                  formatting_func=data_formatter,
                                  max_seq_length=args.max_seq_length,
                                  training_args=training_args)
    
    model.config.use_cache = False

    logger.info(f'Instantiated Trainer')

    # Fine-tune model
    print('Fine-tuning model...')

    trainer.train()

    logger.info(f'Completed fine-tuning')

    # Save adapter weights and tokenizer
    if args.save_model == 'True':

        print('Saving model and tokenizer...')

        trainer.model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)

        logger.info(f'Saved model and tokenizer to {args.save_dir}')

    # Save model to hub
    if args.hub_upload == 'True':

        print('Saving model to hub...')

        trainer.model.push_to_hub(args.hub_save_id, use_auth_token=True)

        logger.info(f'Saved model to hub')

    # Evaluate model on ROUGE, BLEU, and BERTScore
    if args.compute_summarization_metrics == 'True':

        model = trainer.model

        model.eval()
        model.to(args.device)
        model.config.use_cache = True

        print('Evaluating model on ROUGE, BLEU, and BERTScore...')

        metrics = evaluate_hf_model(model, 
                        tokenizer, 
                        data['test'], 
                        input_column=args.input_col,
                        target_column=args.target_col,
                        max_samples=len(data['test']),
                        start_prompt=args.start_prompt,
                        end_prompt=args.end_prompt,)
        
        logger.info(f'Completed ROUGE, BLEU, and BERTScore evaluation')
        wandb.log(metrics)

        # Print metrics
        print('Finetuned Model Metrics:')

        for k, v in metrics.items():
            print(f'{k}: {v}')

    if args.wandb_logging == 'True':
        wandb.finish()