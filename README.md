# Lab-Scale AI

This is the repository housing the code for the Lab-Scale AI project.

## Description

We want to understand whether small, open Laboratory-Scale Language Models (LSLMs?) can compete with large, closed Large Language Models (LLMs), especially on tasks of public or academic interest where the use of transparent, fair, and/or privacy-preserving methods are worth a few percentage points of task performance.

## Usage

Below follow some guidelines for using this repository.

### Text Generation

The generate_from_hf_model.py file allows for text generation from open source models, while the openai_chat_api.py file allows for
chat-based generation with OpenAI models. Open source chat models can be used in chat format by adjusting the prompt appropriately. Text generation is how we expect to assess instruction-tuned language models on the tasks evaluated.

### Fine-Tuning

Currently the finetune_lora_summarization.py file contains functions intended for fine-tuning open-source LLMs, and can be run with a
variety of command line arguments related to data, model quantization & PEFT, logging, and so on. The functions should be trivially
adaptable for tasks other than summarization that can be accomplished using instruction-tuned models. The train_from_recipe.sh file
is intended to allow for a couple of parameters (model, task, quantization) to be passed in such that other parameters can be automatically selected.

```
python finetune_summarization.py
 --model_id tiiuae/falcon-7b-instruct
 --wandb_name falcon-7b-instruct
 --dataset beanham/medsum
 --input_col dialogue
 --target_col section_text
 --train_slice train
 --validation_slice validation
 --test_slice test
 --wandb_logging True
 --max_steps 250
 --start_prompt "### Summarize the following: "
 --end_prompt "### Begin summary: "
```

### Evaluation

The evaluate_summarization.py file contains functions intended for evaluating models on the text summarization task. 

### Library Integrations

This project integrates with three libraries by default. The fine-tuning script logs results to Weights and Biases; text generation with OpenAI models queries GPT-3.5 by default; and pushing HuggingFace models to the Hub requires login (as does downloading gated models). The API keys for these libraries are read from environment variables.

## Questions

1. Can open models running on consumer-grade hardware compete with large, closed models like ChatGPT?
2. What techniques work best to even the playing field? (Quantization, LoRA, Prompt Optimization, Teacher-Student Distillation)
3. Are smaller models tolerably efficient at inference time in comparison to large closed models?
4. Does there exist a disparity with regard to task-relevant fairness metrics between large closed models and small open models?

## Research Design

We will evaluate ~3-7 small, open models against ~3-7 large, closed models. Evaluations will concern 1) performance on tasks of public
or academic interest and 2) fairness in task-relevant settings.

### Model Selection

Open models should be:

1. Mountable (and trainable) on either a consumer-grade 16GB T4 GPU (scenario 1) or a 40GB A100 GPU (scenario 2), with or without quantization.
2. Previously instruction-tuned or chat-tuned to allow more direct comparison with large models and to minimize adaptation costs.
3. Open source or available with one of the following licenses...

Candidate open models:

1. Meta LLAMA-2-Chat 7B
2. Mistral-7B-Instruct
3. Hugging Face H4-Zephyr
4. LMSYS Vicuna
5. Google Flan-T5
6. LAION-AI OpenAssistant
7. TII Falcon
8. Phi-1.5 Microsoft

Closed models should be:

1. Widely ~~hyped~~ used
2. What else?

Candidate closed models:

1. OpenAI GPT-3.5-Turbo
2. OpenAI GPT-4
3. Google Bard/Palm
4. Anthropic Claude
5. Cohere Coral

### Performance task selection

Performance tasks should be:

1. Publicly available with an open, easily accessible dataset and established metrics.
2. Of interest to governmental, non-profit, or academic organizations that may prefer not to rely on closed, pay-per-token models.
3. Accomplishable within the architectural constraints of a smaller model (e.g., article-length summarization would qualify, but book-length summarization would not).

Candidate performance tasks:

1. Fact-Checking: FreshQA (Robert)
2. Text Summarization: CNN DailyNews (Robert)
3. Hybrid Hiring (Isaac)
4. Science QA (Bingbing, PubmedQA, maybe a vision-based science QA model from we have no moat?)
5. Text segmentation alignment (Eva, LLMs for algorithmic work)
6. Entity resolution (Isaac, LLMs to replace custom AI models which replaced classical methods)
7. Text summarization: Medical conversations (Bin)
8. Named entity extraction: Case records extraction (Bingbing's results)
9. Text summarization: nationality & bias (Yiwei) 
 

### Fairness task selection

Fairness tasks should be:

1. Connected to the real-world use of the model (not solely intrinsic).
2. Publicly available with a dataset and evaluation metrics, or at least well-described enough to be reconstructed.

Candidate fairness tasks:

1. Name-nationality bias (Text Summarization)
2. Differential TPR in occupation prediction (Hybrid Hiring)

### Experiment Design

Question: Can $25 models compete with state of the art models on representative tasks?

Principles:
- Each task should be run on all candidate open models above, where possible, AND at least GPT-4 (and maybe other closed models if time)
- Each task should produce a single number at least (Accuracy on test set, etc.)  We will compare this number with that produced by GPT-4.  There is no requirement that these results be comparable across tasks
- Parallel/distributed training is in scope, using well-supported open source frameworks as needed, e.g. HuggingFace
- Each task should use the functions in this repo whenever possible to improve consistency
- Each task should declare whether data is fully public/shareable or not, with a strong preference for public/shareable.
  


### Evaluation Settings

Models can be assessed in the following settings:

1. Zero-shot
    - With CoT prompting
    - With prompt compilation
2. Fine-tuned
    - With low-rank adaptation
3. Distillation
    - With small open models learning from large closed models

Fine-tuning and distillation have a maximum budget of $25 per adaptation and a maximum carbon budget of...

Closed models will likely be assessed only in the zero-shot setting, though we can fine-tune some of them if we decide that is
worthwhile as a point of comparison.

## Results

We can report results from open and closed models on the selected tasks in markdown tables.

Zero-Shot Results by Task

| Task                   | Llama-2-Chat | phi-1.5 | Mistral-7B | Vicuna | GPT-3.5T | GPT-4 |
| ---------------------- | ------------ | ------- | ---------- | ------ | -------- | ----- |
| FreshQA Fact-Checking  |              |         |            |        |          |       |
| CNN Text Summarization |              |         |            |        |          |       |
| Hybrid Hiring          |              |         |            |        |          |       |


Fine-Tuned Results by Task

| Task                   | Llama-2-Chat | phi-1.5 | Mistral-7B | Vicuna | GPT-3.5T | GPT-4 |
| ---------------------- | ------------ | ------- | ---------- | ------ | -------- | ----- |
| FreshQA Fact-Checking  |              |         |            |        |          |       |
| CNN Text Summarization |              |         |            |        |          |       |
| Hybrid Hiring          |              |         |            |        |          |       |


## Overleaf Draft

Draft to be linked after adding some of the basics. We can shoot for ACM FAccT (end of January 2024) or AIES (mid-March 2024).

## Relevant Reading

1. We Have No Moat (leaked Google memo)
2. Llama, Alpaca, Vicuna papers
3. LoRA, qLoRA
4. Substack

(to be linked)
