import json
import argparse
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from typing import Iterable
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from finetune import get_model_and_tokenizer
from evaluate_summarization import evaluate_hf_model
import transformers
import torch
    
#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='True')
    parser.add_argument('--dataset', type=str, default='True')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to mount the model on.')
    args = parser.parse_args()
  
    #-------------------
    # load data
    #-------------------
    print('Getting data...')
    train_data = load_dataset(args.dataset, split='train')
    validation_data = load_dataset(args.dataset, split='validation')
    test_data = load_dataset(args.dataset, split='test')
    
    #-------------------
    # load summarizer
    #-------------------
    print('Getting model and tokenizer...')
    model, tokenizer = get_model_and_tokenizer(args.model,
                                               gradient_checkpointing=False,
                                               device=args.device)
    
    #--------------
    # inference
    #--------------
    model.eval()
    model.to(args.device)
    
    print('Evaluating model on ROUGE, BLEU, and BERTScore...')
    metrics = evaluate_hf_model(model, 
                                tokenizer, 
                                test_data, 
                                input_column='Question',
                                target_column='Sentence',
                                max_samples=len(test_data),
                                start_prompt='### Answer the following question: ',
                                end_prompt='### Begin Answering: ')
    for k, v in metrics.items():
        print(f'{k}: {v}')
      
if __name__ == "__main__":
    main()
