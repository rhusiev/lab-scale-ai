#!/usr/bin/env python3

import numpy as np
import torch
import bitsandbytes as bnb
import logging
import sys
import transformers
import datasets
import argparse
import wandb

from transformers import TrainingArguments
from huggingface_hub import login as hf_login
from os import path, mkdir, getenv
from typing import Mapping

from finetune import format_data_as_instructions, get_model_and_tokenizer, get_lora_model, get_default_trainer, get_dataset_slices
from evaluate_summarization import evaluate_hf_model
from evaluate_qanda import evaluate_hf_model_qa
from evaluate_em import evaluate_hf_model_em

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune a summarization model.')

    # Model ID
    parser.add_argument('--model_id', type=str, default='facebook/opt-125m', help='The model ID to fine-tune.')
    parser.add_argument('--hf_token_var', type=str, default='HF_TOKEN', help='Name of the HuggingFace API token variable name.')
    parser.add_argument('--resume_from_checkpoint', type=str, default='False', help='Whether to resume from a checkpoint.')

    # Device arguments
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to mount the model on.')
    parser.add_argument('--use_mps_device', type=str, default='False', help='Whether to use an MPS device.')
    parser.add_argument('--max_memory', type=str, default='12000MB', help='The maximum memory per GPU, in MB.')

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
    parser.add_argument('--max_steps', type=int, default=500, help='The maximum number of steps to use for fine-tuning.')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='The learning rate to use for fine-tuning.')
    parser.add_argument('--fp16', type=str, default='True', help='Whether to use fp16.')
    parser.add_argument('--output_dir', type=str, default='outputs', help='The directory to save the fine-tuned model.')
    parser.add_argument('--optim', type=str, default='paged_adamw_8bit', help='The optimizer to use for fine-tuning.')

    # Evaluation arguments
    parser.add_argument('--evaluation_strategy', type=str, default='steps', help='The evaluation strategy to use for fine-tuning.')
    parser.add_argument('--eval_steps', type=int, default=25, help='The number of steps between evaluations.')
    parser.add_argument('--eval_on_test', type=str, default='True', help='Whether to evaluate the model on the test set after fine-tuning.')
    parser.add_argument('--compute_summarization_metrics', type=str, default='True', help='Whether to evaluate the model on ROUGE, BLEU, and BERTScore after fine-tuning.')
    parser.add_argument('--compute_qanda_metrics', type=str, default='False', help='Whether to evaluate the model on QA metrics like F1 and Exact Match (from SQUAD).')
    parser.add_argument('--compute_em_metrics', type=str, default='False', help='Whether to evaluate the model on Accuracy, Precision, Recall, and F1.')

    # Hub arguments
    parser.add_argument('--hub_upload', type=str, default='False', help='Whether to upload the model to the hub.')
    parser.add_argument('--hub_save_id', type=str, default='wolferobert3/opt-125m-peft-summarization', help='The name under which the mode will be saved on the hub.')
    parser.add_argument('--save_steps', type=int, default=25, help='The number of steps between saving the model to the hub.')

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
        hf_login(token=getenv(args.hf_token_var))

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
    log_level = args.log_level.upper()
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
            resume_from_checkpoint=args.resume_from_checkpoint == 'True',
            push_to_hub=args.hub_upload == 'True',
            save_steps=args.save_steps,
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

        model_outputs, metrics = evaluate_hf_model(model,
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

        # save model outputs
        np.save(f"{args.model_id.split('/')[1]}_finetuned_model_outputs.npy", model_outputs)

    if args.compute_qanda_metrics == 'True':

        model = trainer.model

        model.eval()
        model.to(args.device)
        model.config.use_cache = True

        print('Evaluating model on QA Metrics (Exact Match and F1 Score)...')

        qa_metrics = evaluate_hf_model_qa(model, 
                        tokenizer, 
                        data['test'], 
                        question_column=args.input_col,
                        answer_column=args.target_col,
                        max_samples=200)
                        #len(data['test']))
        
        logger.info('Completed QA Metrics evaluation')
        wandb.log(qa_metrics)

        # Print metrics
        print('Finetuned Model QA Metrics:')

        for k, v in qa_metrics.items():
            print(f'{k}: {v}')

    if args.compute_em_metrics == 'True':

        model = trainer.model

        model.eval()
        model.to(args.device)
        model.config.use_cache = True

        print('Evaluating model on EM Metrics (F1, Precision, Accuracy, Recall)...')

        em_metrics = evaluate_hf_model_em(model,
                                          tokenizer,
                                          data['test'],
                                          input_column=args.input_col,
                                          target_column=args.target_col,
                                          max_samples=200,
                                          start_prompt=args.start_prompt,
                                          end_prompt=args.end_prompt,)
        # len(data['test']))

        logger.info('Completed EM Metrics evaluation')
        wandb.log(em_metrics)

        # Print metrics
        print('Finetuned Model EM Metrics:')

        for k, v in em_metrics.items():
            print(f'{k}: {v}')

    if args.wandb_logging == 'True':
        wandb.finish()
