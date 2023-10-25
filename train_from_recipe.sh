#!/bin/sh

# Function to print a help message
help()
{
   echo ""
   echo "Usage:$0 -m modelID -t taskID -q quantizationID -conda_env conda_env"
   echo -e "\t-m Model ID"
   echo -e "\t-t Task ID"
   echo -e "\t-q Quantization level"
    echo -e "\t-conda_env Conda environment to activate"
   exit 1
}

# Get user-specified options from the command line
while getopts "m:t:q:conda_env:" opt
do
   case "$opt" in
      m ) modelID="$OPTARG" ;;
      t ) taskID="$OPTARG" ;;
      q ) quantizationID="$OPTARG" ;;
      conda_env ) conda_env="$OPTARG" ;;
      ? ) helpFunction ;;
   esac
done

# Print helpFunction if needed parameters are empty
if [ -z "$modelID" ] || [ -z "$taskID" ] || [ -z "$quantizationID" ]
then
   echo "Please supply a model ID, task ID, and quantization ID";
   help
fi

# Activate conda environment if conda_env argument is passed
if [ "$conda_env" = "deep_learning" ]; then
    echo "Activating conda environment"
    eval "$(conda shell.bash hook)"
    conda activate deep_learning
fi

# Run lora summarization if task_id is 'summarization'
if [ "$taskID" = "summarization" ]; then
    python finetune_lora_summarization.py --model_id $model_id --quantization_type $quantizationID
fi