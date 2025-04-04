# Description: This script trains a model using DPO on the instruction data with preferences.
# Execute: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python -m src.main.py for MPS (MacOS)
# Execute: python -m src.main for CUDA (Linux)
# Update pytorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/mps


import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig
from functools import partial
import copy
import time
from datetime import timedelta
import numpy as np
import random

import src.config as config
from src.dpoLoss import DPOLoss
from src.preferenceDataset import PreferenceDataset
from src.utility import *
from src.trainer import train_model_dpo_simple
# from src.gpuMonitor import log_memory_snapshot
# from src.scheduler import get_scheduler

# Add command-line argument parsing
parser = argparse.ArgumentParser(description='Train a model using DPO with custom hyperparameters')

# DPO loss parameters
parser.add_argument('--beta', type=float, default=config.beta, help='Beta value for DPO loss')
parser.add_argument('--lambda_dpop', type=float, default=config.lambda_dpop, help='Lambda DPOP value')
parser.add_argument('--lambda_kl', type=float, default=config.lambda_kl, help='Lambda KL value')

# Method selection
parser.add_argument('--method', type=int, default=2, help='Method choice (1=dpo, 2=dpop, 3=dpokl, 4=dpopkl)')

# Training parameters
parser.add_argument('--lr', type=float, default=config.learning_rate, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=config.batch_size, help='Batch size')
parser.add_argument('--grad_accum', type=int, default=config.gradient_accumulation_steps, help='Gradient accumulation steps')
parser.add_argument('--epochs', type=int, default=config.num_epochs, help='Number of epochs')
parser.add_argument('--weight_decay', type=float, default=config.weight_decay, help='Weight decay')
parser.add_argument('--max_length', type=int, default=config.allowed_max_length, help='Maximum input length')
parser.add_argument('--max_new_tokens', type=int, default=config.max_new_tokens, help='Maximum tokens to generate')

# Generation parameters
parser.add_argument('--temp', type=float, default=config.temperature, help='Temperature for generation')
parser.add_argument('--top_p', type=float, default=config.top_p, help='Top-p sampling parameter')

# Data parameters
parser.add_argument('--data', type=str, choices=['content', 'structure', 'mixed', 'preference'], default='content', help='Data type to use')

# Evaluation parameters
parser.add_argument('--eval_freq', type=int, default=config.eval_freq, help='Evaluation frequency')
parser.add_argument('--eval_patience', type=int, default=config.early_stopping_patience if hasattr(config, 'early_stopping_patience') else 3, help='Early stopping patience')

args = parser.parse_args()

# ------------------------ Set the model name and cache directory ------------------------
model_workspace_dir = config.model_workspace_dir # directory to save the fine-tuned model
cache_dir = config.cache_dir # cache directory for the Hugging Face model
result_dir = config.result_dir # directory to save the output text and figures
model_name = config.model_name

# Ensure result directory exists
os.makedirs(config.result_dir, exist_ok=True)

# --------- Methods ---------
method_map = {
    1: "dpo",
    2: "dpop",
    3: "dpokl",
    4: "dpopkl"
}

# Override config values with command-line arguments
config.beta = args.beta
config.lambda_dpop = args.lambda_dpop
config.lambda_kl = args.lambda_kl
config.learning_rate = args.lr
config.batch_size = args.batch_size
config.gradient_accumulation_steps = args.grad_accum
config.num_epochs = args.epochs
config.weight_decay = args.weight_decay
config.allowed_max_length = args.max_length
config.max_new_tokens = args.max_new_tokens
config.temperature = args.temp
config.top_p = args.top_p
config.eval_freq = args.eval_freq
if hasattr(config, 'early_stopping_patience'):
    config.early_stopping_patience = args.eval_patience

# Set method choice
method_choice = args.method
method = method_map[method_choice]

# Handle data file selection
data_map = {
    'content': config.file_content,
    'structure': config.file_structure,
    'mixed': config.file_mixed,
    'preference': config.file_preference
}
file_path = data_map[args.data]

# Print the configuration
print(f"\n{'='*50}")
print(f"TRAINING CONFIGURATION:")
print(f"{'='*50}")
print(f"Method: {method.upper()}")
print(f"Data: {args.data}")
print(f"\nDPO Parameters:")
print(f"  Beta: {config.beta}")
if method in ['dpop', 'dpopkl']:
    print(f"  Lambda DPOP: {config.lambda_dpop}")
if method in ['dpokl', 'dpopkl']:
    print(f"  Lambda KL: {config.lambda_kl}")
print(f"\nTraining Parameters:")
print(f"  Learning Rate: {config.learning_rate}")
print(f"  Batch Size: {config.batch_size}")
print(f"  Gradient Accumulation Steps: {config.gradient_accumulation_steps}")
print(f"  Epochs: {config.num_epochs}")
print(f"  Weight Decay: {config.weight_decay}")
print(f"  Evaluation Frequency: {config.eval_freq}")
if hasattr(config, 'early_stopping_patience'):
    print(f"  Early Stopping Patience: {config.early_stopping_patience}")
print(f"\nModel Parameters:")
print(f"  Max Input Length: {config.allowed_max_length}")
print(f"  Max New Tokens: {config.max_new_tokens}")
print(f"  Temperature: {config.temperature}")
print(f"  Top-p: {config.top_p}")
print(f"{'='*50}\n")

# Get output filename from utility function
output_txt = get_output_filename(
    method=method,
    data_file=file_path,
    label="reward_margin",
    learning_rate=config.learning_rate,
    beta=config.beta,
    lambda_dpop=config.lambda_dpop if hasattr(config, 'lambda_dpop') else None,
    lambda_kl=config.lambda_kl if hasattr(config, 'lambda_kl') else None
)
print("Output file path:", output_txt)

# For loss plot
loss_plot_file = get_output_plotname(
    method=method,
    data_file=file_path,
    label="loss",
    learning_rate=config.learning_rate,
    beta=config.beta,
    lambda_dpop=config.lambda_dpop if hasattr(config, 'lambda_dpop') else None,
    lambda_kl=config.lambda_kl if hasattr(config, 'lambda_kl') else None
)
print("Loss plot file path:", loss_plot_file)

# For reward margins plot
margins_plot_file = get_output_plotname(
    method=method,
    data_file=file_path,
    label="reward_margin",
    learning_rate=config.learning_rate,
    beta=config.beta,
    lambda_dpop=config.lambda_dpop if hasattr(config, 'lambda_dpop') else None,
    lambda_kl=config.lambda_kl if hasattr(config, 'lambda_kl') else None
)
print("Reward margins plot file path:", margins_plot_file)

dpo_loss_fn = DPOLoss(beta=config.beta, method=method, lambda_dpop=config.lambda_dpop, lambda_kl=config.lambda_kl)
print(f"Using {method} with beta={config.beta}, lambda_dpop={config.lambda_dpop}, lambda_kl={config.lambda_kl}")

# --------- Device ---------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# --------- Load a Hugging Face model and tokenizer ---------
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16)

# Get the end of text token ID
eot_token_id = tokenizer.eos_token_id  # Instead of tokenizer.convert_tokens_to_ids(eot_token)

policy_model = model # this is the model that will be fine-tuned
ref_model = copy.deepcopy(model) # create a reference model for DPO by copying and freezing the parameters
# log_memory_snapshot("After reference model creation")

for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()
print("Ref model grad status:", next(ref_model.parameters()).requires_grad)
print("Policy model grad status:", next(policy_model.parameters()).requires_grad)

policy_model.to(device)
ref_model.to(device)

# Ensure pad_token is defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Using EOS token '{tokenizer.pad_token}' as PAD token")

print("Model and tokenizer loaded.")

# Load the data
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

print("Number of entries:", len(data))

# Need to use 5-fold cross-validation or more
# Train/val/test split
train_portion = int(len(data) * 0.8)
test_portion = int(len(data) * 0.1) 
val_portion = len(data) - train_portion - test_portion

print("Train portion:", train_portion)
print("Validation portion:", val_portion)
print("Test portion:", test_portion)

# Shuffle the data
random.shuffle(data)

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

customized_collate_fn = partial(
    custom_collate_fn,
    eot_token_id=eot_token_id,
    tokenizer=tokenizer,
    device=device, 
    mask_prompt_tokens=True,  # This is optional
    allowed_max_length=config.allowed_max_length    # The supported context length of the model
)

# Create datasets and dataloaders
train_dataset = PreferenceDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset, 
    batch_size=config.batch_size,
    collate_fn=customized_collate_fn, 
    drop_last=True, 
    shuffle=True
)

val_dataset = PreferenceDataset(val_data, tokenizer)
val_loader = DataLoader(val_dataset, 
    batch_size=config.batch_size, 
    collate_fn=customized_collate_fn, 
    drop_last=False, 
    shuffle=False)

test_dataset = PreferenceDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
)

# print("Train loader:")
# for batch in train_loader:
#     print(
#         batch["chosen"].shape,
#         batch["rejected"].shape,
#     )
# print("\n")

# self-defined stopping criteria
stopping_criteria = StoppingCriteriaList([
    EOTStoppingCriteria(eot_token_id=eot_token_id)
])

# Total steps for the scheduler
# total_steps = num_epochs * len(train_loader) // gradient_accumulation_steps

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader), eta_min=1e-6)

# Scheduler with warmup
# scheduler = get_scheduler(
#     optimizer=optimizer,
#     warmup_steps=config.warmup_steps,
#     total_steps=total_steps
# )

# Evaluate initial state
print("\nEvaluating initial state...")
# log_memory_snapshot("Before initial evaluation")

# Before training loop. If chosen and rejected responses are too similar, the preference margin won’t grow.
batch = next(iter(train_loader))
print("Chosen sample:", tokenizer.decode(batch["chosen"][0].tolist()))
print("Rejected sample:", tokenizer.decode(batch["rejected"][0].tolist()))

res = dpo_loss_fn.evaluate_dpo_loss_loader(
    policy_model=model,
    reference_model=ref_model,
    train_loader=train_loader,
    val_loader=val_loader,
    eval_iter=5
)

# log_memory_snapshot("After initial evaluation")

# Before starting the training, print the initail losses and rewards:
print("Training loss:", res["train_loss"])
print("Validation loss:", res["val_loss"])

print("Train reward margin:", res["train_chosen_reward"] - res["train_rejected_reward"])
print("Val reward margin:", res["val_chosen_reward"] - res["val_rejected_reward"])

print("\n" + "=" * 50)
print("Starting training...")
print("=" * 50)
# log_memory_snapshot("Before training")

start_time = time.time()

torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

tracking = train_model_dpo_simple(
    dpo_loss_fn=dpo_loss_fn,
    optimizer=optimizer,
    scheduler=None,
    policy_model=policy_model,
    reference_model=ref_model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=config.num_epochs,
    eval_freq=config.eval_freq,
    eval_iter=5,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_memory=True
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes (in {str(timedelta(seconds=end_time - start_time))})")

# log_memory_snapshot("After training")

print("Final train/validation statistics:")
print(f"Train loss: {tracking['train_losses'][-1]}")
print(f"Validation loss: {tracking['val_losses'][-1]}")
train_margin = tracking['train_chosen_rewards'][-1] - tracking['train_rejected_rewards'][-1]
val_margin = tracking['val_chosen_rewards'][-1] - tracking['val_rejected_rewards'][-1]
print(f"Train reward margin: {train_margin:.3f}")
print(f"Validation reward margin: {val_margin:.3f}")

# Save the model and tokenizer
save_path = config.fine_tuned_model_path
policy_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved to {save_path}")

# Plot the losses
epochs_tensor = torch.linspace(0, config.num_epochs, len(tracking["train_losses"]))
plot_losses(
    epochs_seen=epochs_tensor,
    tokens_seen=tracking["tokens_seen"],
    train_losses=tracking["train_losses"],
    val_losses=tracking["val_losses"],
    save_path=loss_plot_file,
)

train_reward_margins = [i-j for i,j in zip(tracking["train_chosen_rewards"], tracking["train_rejected_rewards"])]
val_reward_margins = [i-j for i,j in zip(tracking["val_chosen_rewards"], tracking["val_rejected_rewards"])]

plot_losses(
    epochs_seen=epochs_tensor,
    tokens_seen=tracking["tokens_seen"],
    train_losses=train_reward_margins,
    val_losses=val_reward_margins,
    save_path=margins_plot_file
)

fine_tuned_tokenizer = AutoTokenizer.from_pretrained(save_path)
fine_tuned_model = AutoModelForCausalLM.from_pretrained(save_path)
print("Tuned model's tokenizer loaded.")

ref_model.to(device)  # Ensure reference model is on device
fine_tuned_model.to(device)  # Ensure fine-tuned model is on device

print("Starting evaluation...")
# Evaluate the fine-tuned model on the validation set
val_res = dpo_loss_fn.evaluate_dpo_loss_loader(
    policy_model=fine_tuned_model,
    reference_model=ref_model,
    train_loader=None,
    val_loader=val_loader,
    eval_iter=5
)

print("Evaluation loss:", val_res["val_loss"])
print("Evaluation reward margin:", val_res["val_chosen_reward"] - val_res["val_rejected_reward"])

for i, entry in enumerate(val_data[:3]):

    input_text = format_input(entry)

    # Reference Model Generation
    ref_input_ids = text_to_token_ids(input_text, tokenizer).to(device)
    ref_generated = generate(
        model=ref_model,
        idx=ref_input_ids.to(device),
        max_new_tokens=config.max_new_tokens,
        # temperature=temperature,
        # top_p=top_p,
        stopping_criteria=stopping_criteria,
        eot_token_id=eot_token_id
    )
    ref_full_text = tokenizer.decode(ref_generated[0], skip_special_tokens=False)
    ref_response = postprocess_response(ref_full_text)

    # Fine-Tuned Model Generation
    fine_tuned_model_input_ids = text_to_token_ids(input_text, fine_tuned_tokenizer).to(device)
    fine_tuned_model_generated = generate(
        model=fine_tuned_model,
        idx=fine_tuned_model_input_ids.to(device),
        max_new_tokens=config.max_new_tokens,
        # temperature=temperature,
        # top_p=top_p,
        stopping_criteria=stopping_criteria,
        eot_token_id=eot_token_id
    )
    fine_tuned_model_full_text = fine_tuned_tokenizer.decode(fine_tuned_model_generated[0], skip_special_tokens=False)
    fine_tuned_model_response = postprocess_response(fine_tuned_model_full_text)

    # Calculate perplexity
    # ft_perplexity = calculate_perplexity(fine_tuned_model, fine_tuned_tokenizer, input_text)
    # ref_perplexity = calculate_perplexity(ref_model, tokenizer, input_text)

    if ('question' in entry):
        print(f"\nInput{i}: {entry['question']}")
    elif ('instruction' in entry):
        print(f"\nInput{i}: {entry['instruction']}")
    else:
        print(f"\nInput{i}: [No valid input key found]")

    print("\n ----- Reference Model ----- ")
    print(f"Reference Response: {ref_response}")

    print("\n ----- Policy Model ----- ")
    print(f"Policy Response: {fine_tuned_model_response}")

    print("\n ----- Expected Response ----- ")
    print(f"Expected Answer: {entry['chosen']}")

    print("="*80, "\n")

    # Display perplexity
    # print(f"**Fine-Tuned Model Perplexity:** {ft_perplexity:.2f}")
    # print(f"**Original Model Perplexity:** {ref_perplexity:.2f}")
    # print("-" * 80)

# Use the test data to evaluate the fine-tuned model
print("Starting test evaluation...")
test_res = dpo_loss_fn.evaluate_dpo_loss_loader(
    policy_model=fine_tuned_model,
    reference_model=ref_model,
    train_loader=None,
    val_loader=test_loader,
    eval_iter=5
)

print("Test loss:", test_res["val_loss"])
print("Test reward margin:", test_res["val_chosen_reward"] - test_res["val_rejected_reward"])

for i, entry in enumerate(random.sample(test_data[:5], len(test_data[:5]))):
    input_text = format_input(entry)

    # Reference Model Generation
    ref_input_ids = text_to_token_ids(input_text, tokenizer).to(device)
    ref_generated = generate(
        model=ref_model,
        idx=ref_input_ids.to(device),
        max_new_tokens=config.max_new_tokens,
        # temperature=temperature,
        # top_p=top_p,
        stopping_criteria=stopping_criteria,
        eot_token_id=eot_token_id
    )
    ref_full_text = tokenizer.decode(ref_generated[0], skip_special_tokens=False)
    ref_response = postprocess_response(ref_full_text)

    # Fine-Tuned Model Generation
    fine_tuned_model_input_ids = text_to_token_ids(input_text, fine_tuned_tokenizer).to(device)
    fine_tuned_model_generated = generate(
        model=fine_tuned_model,
        idx=fine_tuned_model_input_ids.to(device),
        max_new_tokens=config.max_new_tokens,
        # temperature=temperature,
        # top_p=top_p,
        stopping_criteria=stopping_criteria,
        eot_token_id=eot_token_id
    )
    fine_tuned_model_full_text = fine_tuned_tokenizer.decode(fine_tuned_model_generated[0], skip_special_tokens=False)
    fine_tuned_model_response = postprocess_response(fine_tuned_model_full_text)

    if ('question' in entry):
        print(f"\nInput{i}: {entry['question']}")
    elif ('instruction' in entry):
        print(f"\nInput{i}: {entry['instruction']}")
    else:
        print(f"\nInput{i}: [No valid input key found]")

    print("\n ----- Reference Model ----- ")
    print(f"Reference Response: {ref_response}")

    print("\n ----- Policy Model ----- ")
    print(f"Policy Response: {fine_tuned_model_response}")

    print("\n ----- Expected Response ----- ")
    print(f"Expected Answer: {entry['chosen']}")
    print("="*80, "\n")

    with open(output_txt, "a") as f:
        if ('question' in entry):
            f.write(f"\nInput{i}: {entry['question']}")
        elif ('instruction' in entry):
            f.write(f"\nInput{i}: {entry['instruction']}")
        else:
            f.write(f"\nInput{i}: [No valid input key found]")
        f.write("\n ----- Reference Model ----- ")
        f.write(f"Reference Response: {ref_response}")
        f.write("\n ----- Policy Model ----- ")
        f.write(f"Policy Response: {fine_tuned_model_response}")
        f.write("\n ----- Expected Response ----- ")
        f.write(f"Expected Answer: {entry['chosen']}")
        f.write("="*80 + "\n")
