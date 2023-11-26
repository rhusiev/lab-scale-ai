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

def zeroshot(test_data, pipeline, tokenizer):
    
    #--------------
    # inference
    #--------------
    predictions = []
    gts = []
    for i in tqdm(range(len(test_data))):
        
        system = 'Summarize the following conversation\n'
        dialogue = test_data[i]['dialogue']        
        sequences = pipeline(
            system+dialogue,
            max_length=512,
            do_sample=False,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        predictions.append(sequences[0]['generated_text'])
        gts.append(test_data[i]['section_text'])
    np.save('zeroshot-predictions.npy', predictions)
    np.save('zeroshot-gts.npy', gts)
     
    #results = compute_summarization_metrics(predictions, gts)    
    #with open(f'zeroshot-results.json', 'w') as f:
    #    json.dump(results, f)
    #f.close()

def oneshot(train_data, test_data, pipe):
    #--------------
    # inference
    #--------------
    predictions = []
    gts = []
    for i in tqdm(range(len(test_data))):
        system = 'Summarize the following conversation\n'
        dialogue_example = train_data[0]['dialogue']+'\n'
        summary_example = train_data[0]['section_text']+'\n'
        dialogue = test_data[i]['dialogue']   
        sequences = pipeline(
            system+\dialogue_example+summary_example+dialogue,
            max_length=512,
            do_sample=False,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        predictions.append(sequences[0]['generated_text'])
        gts.append(test_data[i]['section_text'])
    np.save('oneshot-predictions.npy', predictions)
    np.save('oneshot-gts.npy', gts)
  
    #results = compute_summarization_metrics(predictions, gts)    
    #with open('oneshot-results.json', 'w') as f:
    #    json.dump(results, f)
    #f.close()
        
def fewshot(train_data, test_data, pipe):
    #--------------
    # inference
    #--------------
    predictions = []
    gts = []
    for i in tqdm(range(len(test_data))):
        system = 'Summarize the following conversation\n'
        dialogue_example_1 = train_data[0]['dialogue']
        summary_example_1 = train_data[0]['section_text']
        dialogue_example_2 = train_data[1]['dialogue']
        summary_example_2 = train_data[1]['section_text']
        dialogue = test_data[i]['dialogue']
        sequences = pipeline(
            system+\dialogue_example_1+summary_example_1+dialogue_example_2+summary_example_2+dialogue,
            max_length=512,
            do_sample=False,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        predictions.append(sequences[0]['generated_text'])
        gts.append(test_data[i]['section_text'])
    np.save('fewshot-predictions.npy', predictions)
    np.save('fewshot-gts.npy', gts)   
        
    #results = compute_summarization_metrics(predictions, gts)
    #with open('fewshot-results.json', 'w') as f:
    #    json.dump(results, f)
    #f.close()
    
#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='True')
    parser.add_argument('--shottype', type=str, default='True')
    args = parser.parse_args()
    
    #-------------------
    # load data
    #-------------------
    dataset='beanham/medsum'
    train_data = load_dataset(dataset, split='train')
    validation_data = load_dataset(dataset, split='validation')
    test_data = load_dataset(dataset, split='test')
    
    #-------------------
    # load summarizer
    #-------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
        
    #--------------
    # inference
    #--------------
    if args.shottype == 'zero':
        print('Zero Shot...')
        zeroshot(test_data, pipeline, tokenizer)
    elif args.shottype == 'one':
        print('One Shot...')
        oneshot(train_data, test_data, pipeline)
    else:
        print('Few Shot...')
        fewshot(train_data, test_data, pipeline)
    
if __name__ == "__main__":
    main()
