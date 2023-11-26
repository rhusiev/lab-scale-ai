# Recording my experience getting set up.
Below I go through the process of getting setup on labscale-ai. The process should be relatively straightforward, especially if you have experience navigating huggingface.

### 1. Clone Repo
`git clone https://github.com/BillHoweLab/lab-scale-ai.git`

### 2. Install Requirements 
First, create an env using your favorite env manager (I like to use conda, so `conda create -n "labscale" python=3.9`). Note that I used python 3.9 - I think python>=3.9 should be fine.

Then:

```
cd lab-scale-ai
pip install -r requirements.txt
```

### 3. Make sure you have the proper tokens.
You'll need an account with huggingface. You can get a token from the account here: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

You'll also want an account with weights and biases. Then, make a new project, and it will provide you with a token.

### 4. Validate that everything works.
Run `python finetune_summarization.py` with the default arguments. You should be prompted to put in your HF and WandB tokens. If everything was installed properly, this should start to finetune the default opt model. 

This is where you're likely to encounter any environment issues!!

Resolve environment issues on the default finetuning before moving on.

### 5. Grab a large model. 
Once you've validated that the environment and install are all working as expected, you can download one of the huggingface larger models. 

Start with something like Llama, you'll have to download the model weights locally. To do this, you can go somewhere like here: [https://huggingface.co/daryl149/llama-2-7b-chat-hf](https://huggingface.co/daryl149/llama-2-7b-chat-hf) and clone the repository directly into your local branch using `git lfs` (follow the instructions under Clone Repository on the huggingface model page)

*Note*: If you (like me) were running on a remote system that didn't have git lfs installed and you don't have sudo access, you can install git lfs like:

```
wget https://github.com/git-lfs/git-lfs/releases/download/v3.2.0/git-lfs-linux-amd64-v3.2.0.tar.gz
tar -xzf git-lfs-linux-amd64-v3.2.0.tar.gz
PATH=$PATH:/<ABSOLUTE-PATH>/git-lfs-3.2.0/
git lfs install
git lfs version
```

where `<ABSOLUTE-PATH>` is the path to your current directory. 

### 6. Your own task.
Now, to finetune on your own task. 

In my case, I was setting up a version of the MedQA question/answer medical task (based on USMLE exam questions). This lives here now [https://huggingface.co/datasets/lurosenb/medqa](https://huggingface.co/datasets/lurosenb/medqa)

Sometimes, the dataset is not formatted properly for finetuning, so you need to make a slightly modified verison of the dataset.

For example, here I found a version of the dataset I liked. I then downloaded it in a notebook
```
from datasets import load_dataset

dataset = load_dataset("medalpaca/medical_meadow_medqa")
```
And then ran whatever processing I needed (in my case, just to create the proper splits)
```
train_test_split = dataset["train"].train_test_split(test_size=0.4)
test_validation_split = train_test_split["test"].train_test_split(test_size=0.5)

final_dataset = {
    "train": train_test_split["train"],
    "test": test_validation_split["test"],
    "validation": test_validation_split["train"]
}
```
Here, you might have to rename columns, remove NaNs, etc.

Finally, I saved my newly created datasets to individual jsons (apologies for the redundancy). This is to prepare them for upload to huggingface
```
import json

def save_to_json(dataset_split, filename):
    with open(filename, 'w') as file:
        data = [item for item in dataset_split]
        json.dump(data, file, indent=4)

# save each split to a separate json file
save_to_json(final_dataset["train"], 'train_dataset.json')
save_to_json(final_dataset["test"], 'test_dataset.json')
save_to_json(final_dataset["validation"], 'validation_dataset.json')
```

### 7. Run with custom command line arguments.
This is where you are most likely to customize metrics, and may need to customize certain aspects of the finetuning process (hopefully not!).

The arguments for the MedQA finetuning task are:
```
python finetune_summarization.py --model_id llama-2-7b-chat-hf --dataset lurosenb/medqa --input_col input --target_col output  --train_slice train --validation_slice validation --test_slice test --wandb_logging True --wandb_name medqa --max_steps 80 --compute_summarization_metrics False --compute_qanda_metrics True --start_prompt “### Consider the following question with context:” --end_prompt “ ### Please answer with one of the options listed in the brackets:”
```
Note that the `input_col` and `target_col` should match your huggingface dataset column names, and that your filenames should be `train`, `validation` and `test`.

Good luck!