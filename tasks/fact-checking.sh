#!/bin/sh

# Get user-specified options from the command line
while getopts "m:t:s:" opt
do
   case "$opt" in
      m ) modelID="$OPTARG" ;;
      t ) taskID="$OPTARG" ;;
      s ) shots="$OPTARG" ;;
      e ) conda_env="$OPTARG" ;;
   esac
done

# Print helpFunction if needed parameters are empty
if [ -z "$modelID" ] || [ -z "$taskID" ]
then
   modelID=all
   taskID=test
fi

# Set shots to 0 if not specified
if [ -z "$shots" ]
then
   shots=0
fi

# Check if conda environment is specified and activate it
if [ $conda_env ]; then
    echo "Activating conda environment"
    eval "$(conda shell.bash hook)"
    conda activate $conda_env
fi

echo $modelID
echo $taskID
echo $shots

if [ "$modelID" = "gpt-3-5" ] && [ "$taskID" = "all" ]; then

    while [ ! -f /results/gpt-3-5_factcheck_eval_0-shot_metrics.json ]
    do
        python evaluate_factcheck.py \
            --model_type openai \
            --oai_model_id gpt-3.5-turbo \
            --use_model_prompt_defaults openai \
            --results_dir results \
            --run_name gpt-3-5_factcheck_eval \
            --wandb_name gpt-3-5_factcheck_eval \
            --shots 0
        echo "sleeping"
        sleep 300
    done

    while [ ! -f /results/gpt-3-5_factcheck_eval_1-shot_metrics.json ]
    do
        python evaluate_factcheck.py \
            --model_type openai \
            --oai_model_id gpt-3.5-turbo \
            --use_model_prompt_defaults openai \
            --results_dir results \
            --run_name gpt-3-5_factcheck_eval \
            --wandb_name gpt-3-5_factcheck_eval \
            --shots 1
        echo "sleeping"
        sleep 300
    done

    while [ ! -f /results/gpt-3-5_factcheck_eval_2-shot_metrics.json ]
    do
        python evaluate_factcheck.py \
            --model_type openai \
            --oai_model_id gpt-3.5-turbo \
            --use_model_prompt_defaults openai \
            --results_dir results \
            --run_name gpt-3-5_factcheck_eval \
            --wandb_name gpt-3-5_factcheck_eval \
            --shots 2
        echo "sleeping"
        sleep 300
    done

    while [ ! -f /results/gpt-3-5_factcheck_eval_3-shot_metrics.json ]
    do
        python evaluate_factcheck.py \
            --model_type openai \
            --oai_model_id gpt-3.5-turbo \
            --use_model_prompt_defaults openai \
            --results_dir results \
            --run_name gpt-3-5_factcheck_eval \
            --wandb_name gpt-3-5_factcheck_eval \
            --shots 3
        echo "sleeping"
        sleep 300
    done

fi

if [ "$modelID" = "gpt-4" ] && [ "$taskID" = "all" ]; then

    while [ ! -f /results/gpt-4_factcheck_eval_0-shot_metrics.json ]
    do
        python evaluate_factcheck.py \
            --model_type openai \
            --oai_model_id gpt-4 \
            --use_model_prompt_defaults openai \
            --results_dir results \
            --run_name gpt-4_factcheck_eval \
            --wandb_name gpt-4_factcheck_eval \
            --intermediate_outputs_dir gpt-4_factcheck_intermediate \
            --shots 0
        sleep 300
    done

    while [ ! -f /results/gpt-4_factcheck_eval_1-shot_metrics.json ]
    do
        python evaluate_factcheck.py \
            --model_type openai \
            --oai_model_id gpt-4 \
            --use_model_prompt_defaults openai \
            --results_dir results \
            --run_name gpt-4_factcheck_eval \
            --wandb_name gpt-4_factcheck_eval \
            --intermediate_outputs_dir gpt-4_factcheck_intermediate \
            --shots 1
        sleep 300
    done

    while [ ! -f /results/gpt-4_factcheck_eval_2-shot_metrics.json ]
    do
        python evaluate_factcheck.py \
            --model_type openai \
            --oai_model_id gpt-4 \
            --use_model_prompt_defaults openai \
            --results_dir results \
            --run_name gpt-4_factcheck_eval \
            --wandb_name gpt-4_factcheck_eval \
            --intermediate_outputs_dir gpt-4_factcheck_intermediate \
            --shots 2
        sleep 300
    done

    while [ ! -f /results/gpt-4_factcheck_eval_3-shot_metrics.json ]
    do
        python evaluate_factcheck.py \
            --model_type openai \
            --oai_model_id gpt-4 \
            --use_model_prompt_defaults openai \
            --results_dir results \
            --run_name gpt-4_factcheck_eval \
            --wandb_name gpt-4_factcheck_eval \
            --intermediate_outputs_dir gpt-4_factcheck_intermediate \
            --shots 3
        sleep 300
    done

fi

if [ "$modelID" = "gpt-finetuned" ]; then
    python evaluate_factcheck.py \
        --model_type openai \
        --oai_model_id ft:gpt-3.5-turbo-0613:personal::8RIEL9dk \
        --use_model_prompt_defaults openai \
        --results_dir results \
        --run_name gpt-finetuned_factcheck_eval \
        --wandb_name gpt-finetuned_factcheck_eval \
        --system_prompt "You are a helpful assistant specializing in fact-checking." \
        --intermediate_outputs_dir gpt-finetuned_factcheck_intermediate \
        --shots $shots
fi

if [ "$modelID" = "gpt-3-5" ]; then
    python evaluate_factcheck.py \
        --model_type openai \
        --oai_model_id gpt-3.5-turbo \
        --use_model_prompt_defaults openai \
        --results_dir results \
        --run_name gpt-3-5_factcheck_eval \
        --wandb_name gpt-3-5_factcheck_eval \
        --shots $shots
fi

if [ "$modelID" = "gpt-4" ]; then
    python evaluate_factcheck.py \
        --model_type openai \
        --oai_model_id gpt-4 \
        --use_model_prompt_defaults openai \
        --results_dir results \
        --run_name gpt-4_factcheck_eval \
        --wandb_name gpt-4_factcheck_eval \
        --intermediate_outputs_dir gpt-4_factcheck_intermediate \
        --shots $shots
fi

if [ "$modelID" = "gpt-4-turbo" ]; then
    python evaluate_factcheck.py \
        --model_type openai \
        --oai_model_id gpt-4-1106-preview \
        --use_model_prompt_defaults openai \
        --results_dir results \
        --run_name gpt-4-turbo_factcheck_eval \
        --wandb_name gpt-4-turbo_factcheck_eval \
        --intermediate_outputs_dir gpt-4-turbo_factcheck_intermediate \
        --shots $shots
fi

# Run fact-checking fine-tuning depending on the model
if [ "$modelID" = "mistral" ] && [ "$taskID" = "finetune" ]; then
    python finetune_factcheck.py \
        --model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --save_dir mistral_factcheck_final \
        --peft_save_dir mistral_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name mistral_factcheck_finetune \
        --wandb_name mistral_factcheck_finetune \
        --use_model_prompt_defaults mistral \
        --hub_save_id mistral_factcheck_four_bit
fi

if [ "$modelID" = "llama-2-chat" ] && [ "$taskID" = "finetune" ]; then
    python finetune_factcheck.py \
        --model_id meta-llama/Llama-2-7b-chat-hf \
        --save_dir llama-2-chat_factcheck_final \
        --peft_save_dir llama-2-chat_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name llama-2-chat_factcheck_finetune \
        --wandb_name llama-2-chat_factcheck_finetune \
        --use_model_prompt_defaults llama-2 \
        --hub_save_id llama-2-chat_factcheck_four_bit
fi

if [ "$modelID" = "falcon" ] && [ "$taskID" = "finetune" ]; then
    python finetune_factcheck.py \
        --model_id tiiuae/falcon-7b-instruct \
        --save_dir falcon_factcheck_final \
        --peft_save_dir falcon_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name falcon_factcheck_finetune \
        --wandb_name falcon_factcheck_finetune \
        --use_model_prompt_defaults falcon \
        --hub_save_id falcon_factcheck_four_bit
fi

if [ "$modelID" = "mistral" ] && [ "$taskID" = "evaluate" ]; then
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
        --shots $shots
fi

if [ "$modelID" = "llama-2-chat" ] && [ "$taskID" = "evaluate" ]; then
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
        --shots $shots
fi

if [ "$modelID" = "falcon" ] && [ "$taskID" = "evaluate" ]; then
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
        --shots $shots
fi

if [ "$modelID" = "all" ] && [ "$taskID" = "evaluate" ]; then
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
        --shots $shots
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
        --shots $shots
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
        --shots $shots
fi

if [ "$modelID" = "all" ] && [ "$taskID" = "finetune" ]; then
    python finetune_factcheck.py \
        --model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --save_dir mistral_factcheck_final \
        --peft_save_dir mistral_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name mistral_factcheck_finetune \
        --wandb_name mistral_factcheck_finetune \
        --use_model_prompt_defaults mistral \
        --hub_save_id mistral_factcheck_four_bit
    python finetune_factcheck.py \
        --model_id meta-llama/Llama-2-7b-chat-hf \
        --save_dir llama-2-chat_factcheck_final \
        --peft_save_dir llama-2-chat_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name llama-2-chat_factcheck_finetune \
        --wandb_name llama-2-chat_factcheck_finetune \
        --use_model_prompt_defaults llama-2 \
        --hub_save_id llama-2-chat_factcheck_four_bit
    python finetune_factcheck.py \
        --model_id tiiuae/falcon-7b-instruct \
        --save_dir falcon_factcheck_final \
        --peft_save_dir falcon_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name falcon_factcheck_finetune \
        --wandb_name falcon_factcheck_finetune \
        --use_model_prompt_defaults falcon \
        --hub_save_id falcon_factcheck_four_bit
fi

if [ "$modelID" = "all" ] && [ "$taskID" = "evaluate-full" ]; then
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
        --shots 0
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
        --shots 0
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
        --shots 0
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
        --shots 1
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
        --shots 1
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
        --shots 1
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
        --shots 2
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
        --shots 2
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
        --shots 2
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
        --shots 3
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
        --shots 3
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
        --shots 3
fi

if [ "$modelID" = "all" ] && [ "$taskID" = "all" ]; then
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
        --shots 0
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
        --shots 0
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
        --shots 0
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
        --shots 1
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
        --shots 1
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
        --shots 1
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
        --shots 2
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
        --shots 2
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
        --shots 2
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
        --shots 3
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
        --shots 3
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
        --shots 3
    python finetune_factcheck.py \
        --model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --save_dir mistral_factcheck_final \
        --peft_save_dir mistral_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name mistral_factcheck_finetune \
        --wandb_name mistral_factcheck_finetune \
        --use_model_prompt_defaults mistral \
        --hub_save_id mistral_factcheck_four_bit
    python finetune_factcheck.py \
        --model_id meta-llama/Llama-2-7b-chat-hf \
        --save_dir llama-2-chat_factcheck_final \
        --peft_save_dir llama-2-chat_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name llama-2-chat_factcheck_finetune \
        --wandb_name llama-2-chat_factcheck_finetune \
        --use_model_prompt_defaults llama-2 \
        --hub_save_id llama-2-chat_factcheck_four_bit
    python finetune_factcheck.py \
        --model_id tiiuae/falcon-7b-instruct \
        --save_dir falcon_factcheck_final \
        --peft_save_dir falcon_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name falcon_factcheck_finetune \
        --wandb_name falcon_factcheck_finetune \
        --use_model_prompt_defaults falcon \
        --hub_save_id falcon_factcheck_four_bit
fi

if [ "$modelID" = "all" ] && [ "$taskID" = "test" ]; then
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
    	--split test[:50] \
        --shots 0
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
    	--split test[:50] \
        --shots 0
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
    	--split test[:50] \
        --shots 0
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
    	--split test[:50] \
        --shots 3
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
    	--split test[:50] \
        --shots 3
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
    	--split test[:50] \
        --shots 3
    python finetune_factcheck.py \
        --model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --save_dir mistral_factcheck_final \
        --peft_save_dir mistral_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name mistral_factcheck_finetune \
        --wandb_name mistral_factcheck_finetune \
        --use_model_prompt_defaults mistral \
        --hub_save_id mistral_factcheck_four_bit-test \
        --train_slice train[:50] \
        --test_slice test[:50]
    python finetune_factcheck.py \
        --model_id meta-llama/Llama-2-7b-chat-hf \
        --save_dir llama-2-chat_factcheck_final \
        --peft_save_dir llama-2-chat_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name llama-2-chat_factcheck_finetune \
        --wandb_name llama-2-chat_factcheck_finetune \
        --use_model_prompt_defaults llama-2 \
        --hub_save_id llama-2-chat_factcheck_four_bit-test \
        --train_slice train[:50] \
        --test_slice test[:50]
    python finetune_factcheck.py \
        --model_id tiiuae/falcon-7b-instruct \
        --save_dir falcon_factcheck_final \
        --peft_save_dir falcon_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name falcon_factcheck_finetune \
        --wandb_name falcon_factcheck_finetune \
        --use_model_prompt_defaults falcon \
        --hub_save_id falcon_factcheck_four_bit-test \
        --train_slice train[:50] \
        --test_slice test[:50]
fi

if [ "$modelID" = "all" ] && [ "$taskID" = "test-eval" ]; then
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
    	--split test[:50] \
        --shots 0
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
    	--split test[:50] \
        --shots 0
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
    	--split test[:50] \
        --shots 0
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
    	--split test[:50] \
        --shots 3
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
    	--split test[:50] \
        --shots 3
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
    	--split test[:50] \
        --shots 3
fi

if [ "$modelID" = "all" ] && [ "$taskID" = "test-finetune" ]; then
    python finetune_factcheck.py \
        --model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --save_dir mistral_factcheck_final \
        --peft_save_dir mistral_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name mistral_factcheck_finetune \
        --wandb_name mistral_factcheck_finetune \
        --use_model_prompt_defaults mistral \
        --hub_save_id mistral_factcheck_four_bit-test \
        --train_slice train[:50] \
        --test_slice test[:50]
    python finetune_factcheck.py \
        --model_id meta-llama/Llama-2-7b-chat-hf \
        --save_dir llama-2-chat_factcheck_final \
        --peft_save_dir llama-2-chat_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name llama-2-chat_factcheck_finetune \
        --wandb_name llama-2-chat_factcheck_finetune \
        --use_model_prompt_defaults llama-2 \
        --hub_save_id llama-2-chat_factcheck_four_bit-test \
        --train_slice train[:50] \
        --test_slice test[:50]
    python finetune_factcheck.py \
        --model_id tiiuae/falcon-7b-instruct \
        --save_dir falcon_factcheck_final \
        --peft_save_dir falcon_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name falcon_factcheck_finetune \
        --wandb_name falcon_factcheck_finetune \
        --use_model_prompt_defaults falcon \
        --hub_save_id falcon_factcheck_four_bit-test \
        --train_slice train[:50] \
        --test_slice test[:50]
fi

if [ "$modelID" = "mistral" ] && [ "$taskID" = "all" ]; then
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
        --shots 0
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
        --shots 1
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
        --shots 2
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --use_model_prompt_defaults mistral \
        --results_dir results \
        --run_name mistral_factcheck_eval \
        --wandb_name mistral_factcheck_eval \
        --shots 3
    python finetune_factcheck.py \
        --model_id mistralai/Mistral-7B-Instruct-v0.1 \
        --save_dir mistral_factcheck_final \
        --peft_save_dir mistral_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name mistral_factcheck_finetune \
        --wandb_name mistral_factcheck_finetune \
        --use_model_prompt_defaults mistral \
        --hub_save_id mistral_factcheck_four_bit
fi

if [ "$modelID" = "llama" ] && [ "$taskID" = "all" ]; then
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
        --shots 0
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
        --shots 1
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
        --shots 2
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id meta-llama/Llama-2-7b-chat-hf \
        --use_model_prompt_defaults llama-2 \
        --results_dir results \
        --run_name llama-2-chat_factcheck_eval \
        --wandb_name llama-2-chat_factcheck_eval \
        --shots 3
    python finetune_factcheck.py \
        --model_id meta-llama/Llama-2-7b-chat-hf \
        --save_dir llama-2-chat_factcheck_final \
        --peft_save_dir llama-2-chat_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name llama-2-chat_factcheck_finetune \
        --wandb_name llama-2-chat_factcheck_finetune \
        --use_model_prompt_defaults llama-2 \
        --hub_save_id llama-2-chat_factcheck_four_bit
fi

if [ "$modelID" = "falcon" ] && [ "$taskID" = "all" ]; then
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
        --shots 0
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
        --shots 1
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
        --shots 2
    python evaluate_factcheck.py \
        --model_type hf \
        --hf_model_id tiiuae/falcon-7b-instruct \
        --use_model_prompt_defaults falcon \
        --results_dir results \
        --run_name falcon_factcheck_eval \
        --wandb_name falcon_factcheck_eval \
        --shots 3
    python finetune_factcheck.py \
        --model_id tiiuae/falcon-7b-instruct \
        --save_dir falcon_factcheck_final \
        --peft_save_dir falcon_factcheck_peft \
        --results_dir results \
        --log_dir logs \
        --run_name falcon_factcheck_finetune \
        --wandb_name falcon_factcheck_finetune \
        --use_model_prompt_defaults falcon \
        --hub_save_id falcon_factcheck_four_bit
fi