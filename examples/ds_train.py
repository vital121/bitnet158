"""
This script fine-tunes a pre-trained causal language model (CLM) using the Hugging Face Transformers library on a specified dataset.
It supports injecting custom linear layers (`BitLinear` or `BitLinear158`) into the model architecture, which can be useful for
experimental modifications or optimizations. Additionally, it provides functionality for data loading, tokenization,
training argument configuration, and custom callbacks for monitoring training progress.

Usage:
    Run this script from the command line with the necessary arguments. For example:
    ```
    python script_name.py --model_id microsoft/phi-2 --dataset EleutherAI/wikitext_document_level --subset wikitext-103-raw-v1 --output_dir saved_model --inject BitLinear
    ```

Arguments:
    --model_id (str): The ID of the pre-trained model to fine-tune. Defaults to "microsoft/phi-2".
    --dataset (str): The dataset to use for fine-tuning. Defaults to "EleutherAI/wikitext_document_level".
    --subset (str): The specific subset of the dataset to use. Defaults to "wikitext-103-raw-v1".
    --output_dir (str): Directory where the fine-tuned model and checkpoints will be saved. Defaults to "saved_model".
    --inject (str): Specifies the type of custom linear layer to inject into the model. Choices are "BitLinear158", "BitLinear", and "None". Defaults to "BitLinear".

DeepSpeed Arguments:
    --train_args_file (str): Optional file containing training arguments for advanced configurations.
    --deepspeed (str): Optional DeepSpeed configuration file for leveraging DeepSpeed optimizations.
    --local_rank (int): Local rank passed from distributed launcher. Default is -1, indicating non-distributed training.

Functionality:
    1. Loads the specified pre-trained causal language model and tokenizer.
    2. Optionally modifies the model by replacing `nn.Linear` layers with custom `BitLinear` or `BitLinear158` layers.
    3. Loads and tokenizes the specified dataset, preparing it for training.
    4. Configures training arguments, including DeepSpeed integration if specified.
    5. Initializes a `Trainer` instance with the model, data, tokenizer, and training arguments.
    6. Starts the training process with an optional custom callback for monitoring first layer gradients.

Custom Callbacks:
    - PrintFirstLayerGradientsCallback: A callback that prints the gradients of the first layer of the model at the end of each training step.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from bitnet158 import BitLinear, BitLinear158, inject
import argparse
from transformers import TrainerCallback

### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="microsoft/phi-2")
parser.add_argument("--dataset", type=str, default="EleutherAI/wikitext_document_level")
parser.add_argument("--subset", type=str, default="wikitext-103-raw-v1")
parser.add_argument("--output_dir", type=str, default="saved_model")
parser.add_argument("--inject", choices=["BitLinear158", "BitLinear", "None"], default="BitLinear")

# DeepSpeed Arguments
parser.add_argument("--train_args_file", type=str, default='--', help="")
parser.add_argument("--deepspeed", type=str, default='--', help="")
parser.add_argument('--local_rank', type=int, default=-1,
                help='local rank passed from distributed launcher')

args = parser.parse_args()

### Load Model
model_id = args.model_id

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)

# print number of parameters that are trainable
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters: ", num_params)

### Load Dataset
ds = load_dataset(args.dataset, args.subset, split="train")
val_ds = load_dataset(args.dataset, args.subset, split="test")
def tokenize_function(examples):
    return tokenizer(examples["page"], truncation=True, max_length=256)

tokenized_datasets = ds.map(tokenize_function, batched=False, num_proc=32, remove_columns=["page"])
tokenized_datasets_val = val_ds.map(tokenize_function, batched=False, num_proc=32, remove_columns=["page"])
print(tokenized_datasets)

### Inject BitLinear layers
if args.inject == "BitLinear158":
    inject(model, copy_weights=True, module_class=BitLinear158)
elif args.inject == "BitLinear":
    inject(model, copy_weights=True, module_class=BitLinear)
else:
    pass


class PrintFirstLayerGradientsCallback(TrainerCallback):
    """
    A custom callback that prints the gradient of the first layer of the model at each training step.
    """
    def on_step_end(self, args, state, control, **kwargs):
        # Assuming 'model' is your model instance and it's a PyTorch model
        # You may need to adjust the layer name depending on your model architecture
        num_params = 0
        print("First layer gradients")
        for k, v in model.named_parameters():
            if v.grad is not None:
                print(k, v.grad)
                num_params += v.numel()
            else:
                print(k, "None")
        print("Number of parameters: ", num_params)


### Start Training
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        deepspeed="ds_config.json",
        output_dir=args.output_dir,
        save_steps=100,
        fp16=True,
        save_total_limit=3,
        per_device_train_batch_size=2,
        eval_steps=100,
        evaluation_strategy="steps",
        logging_steps=10,
        learning_rate=1e-3,
    ),
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[PrintFirstLayerGradientsCallback()],
)

trainer.train()
