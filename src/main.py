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
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from functools import partial
import copy
import time
from datetime import timedelta
import numpy as np
import random

import src.config as config
from src.dpoLoss import DPOLoss
from src.preferenceDataset import PreferenceDataset
from src.utils import *
from src.trainer import train_model
from src.argsParse import *

# ----------------------------------- Argument Parsing -----------------------------------
args = parse_args() # Parse command-line arguments
update_config_from_args(args) # Update config with parsed arguments
print_configuration() # Print the configuration

# Get relevant parameters for the selected method
dpo_params = get_dpo_params(config.method_name)

# Initialize DPO loss function with only the relevant parameters
dpo_loss_fn = DPOLoss(method=config.method_name, **dpo_params)

# Print the parameters being used for clarity
param_str = ", ".join([f"{k}={v}" for k, v in dpo_params.items()])
print(f"Using {config.method_name} with {param_str}")

# ------------------------------- Set the cache directory -------------------------------
model_workspace_dir = config.model_workspace_dir # directory to save the fine-tuned model
cache_dir = config.cache_dir # cache directory for the Hugging Face model
result_dir = config.result_dir # directory to save the output text and figures

# ---------------------------- Ensure result directory exists ----------------------------
os.makedirs(config.result_dir, exist_ok=True)

# ----------------------- Get each filename from utils function ------------------------ 
# For output text
output_json = get_output_filename(
    model=config.model_name,
    method=config.method_name,
    file=config.training_data_filename,
    label="generated_output",
    learning_rate=config.learning_rate,
    beta=config.beta,
    lambda_dpop=config.lambda_dpop if hasattr(config, 'lambda_dpop') else None,
    lambda_shift=config.lambda_shift if hasattr(config, 'lambda_shift') else None,
    typename="json" # Specify the file type
)
print("Output file path:", output_json)

# For loss plot
loss_plot_file = get_output_filename(
    model=config.model_name,
    method=config.method_name,
    file=config.training_data_filename,
    label="loss",
    learning_rate=config.learning_rate,
    beta=config.beta,
    lambda_dpop=config.lambda_dpop if hasattr(config, 'lambda_dpop') else None,
    lambda_shift=config.lambda_shift if hasattr(config, 'lambda_shift') else None,
    typename="png" # Specify the file type
)
print("Loss plot file path:", loss_plot_file)

# For reward margins plot
margins_plot_file = get_output_filename(
    model=config.model_name,
    method=config.method_name,
    file=config.training_data_filename,
    label="reward_margin",
    learning_rate=config.learning_rate,
    beta=config.beta,
    lambda_dpop=config.lambda_dpop if hasattr(config, 'lambda_dpop') else None,
    lambda_shift=config.lambda_shift if hasattr(config, 'lambda_shift') else None,
    typename="png" # Specify the file type
)
print("Reward margins plot file path:", margins_plot_file)

# ---------------------------------------- Device ----------------------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# ----------------------- Load a Hugging Face model and tokenizer ------------------------
tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(config.model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16)

eos_token_id = tokenizer.eos_token_id # Get the end of text token ID

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

# --------------------------- Set the tokenizer's padding token --------------------------
tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
model.config.pad_token_id = tokenizer.pad_token_id # updating model config
tokenizer.padding_side = 'right' # padding to right (prevent showing warning)
print(f"Set PAD token to '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
print(f"Using EOS token '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")

print("Model and tokenizer loaded.")

# ------------------------------------- Load the data ------------------------------------
try:
    with open(config.training_data_filename, "r", encoding="utf-8") as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"File {config.training_data_filename} not found. Please check the path.")
    exit(1)

print("Number of entries:", len(data))

# ----------------------------- data set pre-processing and shuffling -----------------------------
# Randomly select 10% of the data for testing which is fixed data in multiple runs
random.seed(42)
test_size = int(len(data) * 0.1)
test_data = random.sample(data, test_size)

# Remove test data from the original dataset
remaining = [d for d in data if d not in test_data]

# Randomly shuffle the remaining data
random.shuffle(remaining)

# Split the remaining data into training and validation sets
train_size = int(len(data) * 0.8)
train_data = remaining[:train_size]
val_data   = remaining[train_size:]

print("Train size:", train_size)
print("Validation size:", len(remaining) - train_size)
print("Test size:", test_size)

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

# ------------------------------------------------ Set warmup steps ------------------------------------------------
# Compute the number of training steps
batches_per_epoch = train_size // config.batch_size
optimization_steps_per_epoch = batches_per_epoch // config.gradient_accumulation_steps
num_training_steps = optimization_steps_per_epoch * config.num_epochs

# Dynamic warmup steps
num_warmup_steps = int(0.1 * num_training_steps)
print(f"Dataset size: {len(data)}, num_training_steps: {num_training_steps}, num_warmup_steps: {num_warmup_steps}")
# ------------------------------------------------------------------------------------------------------------------

# ---------------------------- Custom collate function for DataLoader ---------------------------
customized_collate_fn = partial(
    custom_collate_fn,
    eos_token_id=eos_token_id,
    tokenizer=tokenizer,
    device=device, 
    mask_prompt_tokens=True,  # This is optional
    allowed_max_length=config.allowed_max_length    # The supported context length of the model
)

# ---------------------------- Create datasets and dataloaders ---------------------------
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

print ("\n" + "=" * 50)
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
        eos_token_id=eos_token_id
    )
    ref_full_text = tokenizer.decode(ref_generated[0], skip_special_tokens=False)
    ref_response = postprocess_response(ref_full_text)


    if ('question' in entry):
        print(f"\nInput{i}: {entry['question']}")
    elif ('instruction' in entry):
        print(f"\nInput{i}: {entry['instruction']}")
    else:
        print(f"\nInput{i}: [No valid input key found]")

    print("\n ----- Reference Model ----- ")
    print(f"Reference Response: {ref_response}")

    print("\n ----- Expected Response ----- ")
    print(f"Expected Answer: {entry['chosen']}")

    print("="*80, "\n")

print("\n" + "=" * 50)
print("Starting training...")
print("=" * 50)
# log_memory_snapshot("Before training")

# Initialize the optimizer and scheduler
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

start_time = time.time()

torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

tracking = train_model(
    dpo_loss_fn=dpo_loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    policy_model=policy_model,
    reference_model=ref_model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=config.num_epochs,
    eval_freq=config.eval_freq,
    eval_iter=5,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    tokenizer=tokenizer,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes (in {str(timedelta(seconds=end_time - start_time))})")
print(f" with {config.method_name}, {config.training_data_filename}, {config.model_name}, "
      f"lr={config.learning_rate}, beta={config.beta}, lambda_dpop={config.lambda_dpop}, lambda_shift={config.lambda_shift}")

# log_memory_snapshot("After training")

print("Final train/validation statistics:")
print(f"Train loss: {tracking['train_losses'][-1]}")
print(f"Validation loss: {tracking['val_losses'][-1]}")
train_margin = tracking['train_chosen_rewards'][-1] - tracking['train_rejected_rewards'][-1]
val_margin = tracking['val_chosen_rewards'][-1] - tracking['val_rejected_rewards'][-1]
print(f"Train reward margin: {train_margin:.3f}")
print(f"Validation reward margin: {val_margin:.3f}")
print(f"Tokens seen: {tracking['tokens_seen'][-1]}")

log_final_result_csv(method=config.method_name,
                     file=config.training_data_filename,
                     epoch=config.num_epochs,
                     beta=config.beta,
                     lambda_dpop=config.lambda_dpop if hasattr(config, 'lambda_dpop') else None,
                     lambda_shift=config.lambda_shift if hasattr(config, 'lambda_shift') else None,
                     learning_rate=config.learning_rate,
                     train_loss=tracking['train_losses'][-1],
                     val_loss=tracking['val_losses'][-1],
                     train_reward_margin=train_margin,
                     val_reward_margin=val_margin
                     )

print("\nAnalyzing batch records for significant loss changes:")
if "batch_records" in tracking and tracking["batch_records"]:
    # Find batches with the largest loss increases and decreases
    sorted_records = sorted(tracking["batch_records"], key=lambda x: x["loss_change"])
    
    # Top 3 decreases (improvements)
    print("\nTop 3 Loss Decreases (Improvements):")
    for record in sorted_records[:3]:
        print(f"Batch {record['batch_idx']} - Loss change: {record['loss_change']:.4f}")
        print(f"Reward difference: {record['reward_diff']:.4f}")
        
    # Top 3 increases (deteriorations)
    print("\nTop 3 Loss Increases (Deteriorations):")
    for record in sorted_records[-3:]:
        print(f"Batch {record['batch_idx']} - Loss change: {record['loss_change']:.4f}")
        print(f"Reward difference: {record['reward_diff']:.4f}")
else:
    print("No batch records found in tracking data")

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
    label="loss"
)

train_reward_margins = [i-j for i,j in zip(tracking["train_chosen_rewards"], tracking["train_rejected_rewards"])]
val_reward_margins = [i-j for i,j in zip(tracking["val_chosen_rewards"], tracking["val_rejected_rewards"])]

plot_losses(
    epochs_seen=epochs_tensor,
    tokens_seen=tracking["tokens_seen"],
    train_losses=train_reward_margins,
    val_losses=val_reward_margins,
    save_path=margins_plot_file,
    label="reward margin"
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
        eos_token_id=eos_token_id
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
        eos_token_id=eos_token_id
    )
    fine_tuned_model_full_text = fine_tuned_tokenizer.decode(fine_tuned_model_generated[0], skip_special_tokens=False)
    fine_tuned_model_response = postprocess_response(fine_tuned_model_full_text)

    # Calculate perplexity for both models
    ref_perplexity = calculate_perplexity(
        model=ref_model,
        tokenizer=tokenizer,
        texts=input_text,
        max_length=config.allowed_max_length,
        device=device
    )
    ft_perplexity = calculate_perplexity(
        model=fine_tuned_model,
        tokenizer=fine_tuned_tokenizer,
        texts=input_text,
        max_length=config.allowed_max_length,
        device=device
    )

    if ('question' in entry):
        print(f"\nInput{i}: {entry['question']}")
    elif ('instruction' in entry):
        print(f"\nInput{i}: {entry['instruction']}")
    else:
        print(f"\nInput{i}: [No valid input key found]")

    print("\n ----- Reference Model ----- ")
    print(f"Reference Response: {ref_response}")
    print(f"Perplexity: {ref_perplexity:.2f}")

    print("\n ----- Policy Model ----- ")
    print(f"Policy Response: {fine_tuned_model_response}")
    print(f"Perplexity: {ft_perplexity:.2f}")

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

# List for storing the test results to write to the json file
test_results = []

# Check first entry to determine data type
input_key = "question" if "question" in test_data[0] else "instruction"

# Check the maximum sequence length in the test data
max_length = 0
for entry in test_data:
    tokens = fine_tuned_tokenizer(format_input(entry), add_special_tokens=True).input_ids
    max_length = max(max_length, len(tokens))
print(f"Test data max sequence length: {max_length}")

# Set stride based on the maximum length
if max_length > config.allowed_max_length:
    print("Warning: Long sequences detected, using stride=512")
    stride = 512
else:
    stride = None

try:
    for i, entry in enumerate(test_data):

        input_text = format_input(entry)

        # Reference Model Generation
        ref_input_ids = text_to_token_ids(input_text, tokenizer).to(device)
        ref_generated = generate(
            model=ref_model,
            idx=ref_input_ids.to(device),
            max_new_tokens=config.max_new_tokens,
            # temperature=temperature,
            # top_p=top_p,
            eos_token_id=eos_token_id
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
            eos_token_id=eos_token_id
        )
        fine_tuned_model_full_text = fine_tuned_tokenizer.decode(fine_tuned_model_generated[0], skip_special_tokens=False)
        fine_tuned_model_response = postprocess_response(fine_tuned_model_full_text)
        
        # Calculate perplexity
        ref_perplexity = calculate_perplexity(
            model=ref_model,
            tokenizer=tokenizer,
            texts=input_text,
            max_length=config.allowed_max_length,
            stride=stride,
            device=device
        )

        ft_perplexity = calculate_perplexity(
            model=fine_tuned_model,
            tokenizer=fine_tuned_tokenizer,
            texts=input_text,
            max_length=config.allowed_max_length,
            stride=stride,
            device=device
        )

        # Use the previously determined input key
        print(f"\nInput {i}:\n {entry[input_key]}")
            
        print("\n ----- Reference Model ----- ")
        print(f"Reference Response: {ref_response}")
        print(f"Perplexity: {ref_perplexity:.2f}")

        print("\n ----- Policy Model ----- ")
        print(f"Policy Response: {fine_tuned_model_response}")
        print(f"Perplexity: {ft_perplexity:.2f}")

        print("\n ----- Expected Response ----- ")
        print(f"Expected Answer: {entry['chosen']}")
        print("="*80, "\n")

        # Create a single sample object and append to the results list
        sample = {
            input_key: entry[input_key],
            "ref_response": ref_response,
            "policy_response": fine_tuned_model_response,
            "expected_response": entry['chosen'],
            "ref_perplexity": ref_perplexity,
            "policy_perplexity": ft_perplexity
        }
        test_results.append(sample)

except KeyboardInterrupt:
    print("\nInterrupted! Saving partial results...")

finally:
    # Save the test results to a JSON file
    with open(output_json, "w") as f:
        json.dump(test_results, f, indent=4)
    print("Test results saved to:", output_json)