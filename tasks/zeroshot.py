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
import transformers
import torch


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
    parser.add_argument('--shottype', type=str, default='True')
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
    model, tokenizer = get_model_and_tokenizer(args.model,gradient_checkpointing=False)
    
    #--------------
    # inference
    #--------------
    model.eval()
    print('Evaluating model on ROUGE, BLEU, and BERTScore...')
    metrics = evaluate_hf_model(model, 
                                tokenizer, 
                                test_data, 
                                input_column='Question',
                                target_column='Sentence',
                                max_samples=len(test_data),
                                start_prompt='Answer the following question:')
    
if __name__ == "__main__":
    main()
