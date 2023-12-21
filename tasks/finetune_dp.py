import torch
import bitsandbytes as bnb

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling, AutoModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
from tqdm import tqdm
from typing import Mapping, Iterable

import dp_transformers
from dp_transformers import PrivacyArguments
from dp_transformers.grad_sample.transformers import conv_1d

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
        fp16=False, 
        logging_steps=1,
        output_dir='outputs',
        optim='paged_adamw_8bit' if torch.cuda.is_available() else 'adamw_torch',
        use_mps_device=False,
        log_level='info',
        logging_first_step=True,
        evaluation_strategy='steps',
        eval_steps=25
    )

# NOTE: Setting a high epsilon value here to help debug numerical instability issues
DEFAULT_PRIVACY_ARGS = PrivacyArguments(
        target_epsilon=50,
        per_sample_max_grad_norm=1.2
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
                            device: str='auto') -> tuple[AutoModel, AutoTokenizer]:
    """
    Returns a Transformers model and tokenizer for fine-tuning. If quantization_type is provided, the model will be quantized and prepared for training.
    """

    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='right')

    # Set the pad token (needed for trainer class, no value by default for most causal models)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Download the model, quantize if requested
    if quantization_type:
        model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                     quantization_config=QUANZATION_MAP[quantization_type], 
                                                     device_map=device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                     device_map=device)

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
                training_args: TrainingArguments=None,
                privacy_args: PrivacyArguments=None) -> SFTTrainer:
    """
    Returns the default trainer for fine-tuning a summarization model based on the specified training config.
    """
    data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)

    # NOTE: OpacusDPTrainer by default wraps Trainer, not SFTTrainer. 
    # This means some very minor modifications to the dp_transformers class.

    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=training_args if training_args else DEFAULT_TRAINING_ARGS,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        data_collator=data_collator,
        privacy_args=privacy_args if privacy_args else DEFAULT_PRIVACY_ARGS,
        max_seq_length=max_seq_length,
        packing=False
    )

    return trainer