# Description: This script trains a model using DPO on the instruction data with preferences.
# Execute: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python -m src.main.py for MPS (MacOS)
# Execute: python -m src.main for CUDA (Linux)
# Update pytorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/mps


import os
import json
import torch
from torch.utils.data import DataLoader
# from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig
from functools import partial
import copy
import time
from datetime import timedelta
import numpy as np
import multiprocessing

import src.config as config
from src.dpoLoss import DPOLoss
from src.preferenceDataset import PreferenceDataset
from src.utility import *
from src.trainer import train_model_dpo_simple
from src.scheduler import get_scheduler

# --------- File Paths ---------
model_workspace_dir = config.model_workspace_dir # directory to save the fine-tuned model
cache_dir = config.cache_dir # cache directory for the Hugging Face model
result_dir = config.result_dir # directory to save the output text and figures
model_name = config.model_name
file_path = config.file_content

# --------- Hyperparameters ---------
allowed_max_length = config.allowed_max_length
max_new_tokens = config.max_new_tokens
batch_size = config.batch_size
gradient_accumulation_steps = config.gradient_accumulation_steps
num_epochs = config.num_epochs
learning_rate = config.learning_rate
weight_decay = config.weight_decay
temperature = config.temperature
top_p = config.top_p
dpo_loss_fn = DPOLoss(beta=config.beta, lambda_kl=config.lambda_kl)

# --------- Device ---------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

# --------- Load a Hugging Face model and tokenizer ---------
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16)

# Get the end of text token ID
eot_token_id = tokenizer.eos_token_id  # Instead of tokenizer.convert_tokens_to_ids(eot_token)

policy_model = model # this is the model that will be fine-tuned
ref_model = copy.deepcopy(model) # create a reference model for DPO by copying and freezing the parameters

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
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1) 
val_portion = len(data) - train_portion - test_portion

print("Train portion:", train_portion)
print("Validation portion:", val_portion)
print("Test portion:", test_portion)

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
    allowed_max_length=allowed_max_length    # The supported context length of the model
)

# Create datasets and dataloaders
train_dataset = PreferenceDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    collate_fn=customized_collate_fn, 
    drop_last=True, 
    shuffle=True
)

val_dataset = PreferenceDataset(val_data, tokenizer)
val_loader = DataLoader(val_dataset, 
    batch_size=batch_size, 
    collate_fn=customized_collate_fn, 
    drop_last=False, 
    shuffle=False)

test_dataset = PreferenceDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
)

print("Train loader:")
for batch in train_loader:
    print(
        batch["chosen"].shape,
        batch["rejected"].shape,
    )
print("\n")

# self-defined stopping criteria
stopping_criteria = StoppingCriteriaList([
    EOTStoppingCriteria(eot_token_id=eot_token_id)
])

# Total steps for the scheduler
total_steps = num_epochs * len(train_loader) // gradient_accumulation_steps

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader), eta_min=1e-6)

# Scheduler with warmup
scheduler = get_scheduler(
    optimizer=optimizer,
    warmup_steps=config.warmup_steps,
    total_steps=total_steps
)

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

# Before starting the training, print the initail losses and rewards:
print("Training loss:", res["train_loss"])
print("Validation loss:", res["val_loss"])

print("Train reward margin:", res["train_chosen_reward"] - res["train_rejected_reward"])
print("Val reward margin:", res["val_chosen_reward"] - res["val_rejected_reward"])

start_time = time.time()

torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

tracking = train_model_dpo_simple(
    dpo_loss_fn=dpo_loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    policy_model=policy_model,
    reference_model=ref_model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=num_epochs,
    eval_freq=config.eval_freq,
    eval_iter=5,
    gradient_accumulation_steps=gradient_accumulation_steps
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes (in {str(timedelta(seconds=end_time - start_time))})")

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
epochs_tensor = torch.linspace(0, num_epochs, len(tracking["train_losses"]))
plot_losses(
    epochs_seen=epochs_tensor,
    tokens_seen=tracking["tokens_seen"],
    train_losses=tracking["train_losses"],
    val_losses=tracking["val_losses"],
    label="DPO_loss"
)

train_reward_margins = [i-j for i,j in zip(tracking["train_chosen_rewards"], tracking["train_rejected_rewards"])]
val_reward_margins = [i-j for i,j in zip(tracking["val_chosen_rewards"], tracking["val_rejected_rewards"])]

plot_losses(
    epochs_seen=epochs_tensor,
    tokens_seen=tracking["tokens_seen"],
    train_losses=train_reward_margins,
    val_losses=val_reward_margins,
    label="reward_margins"
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
        max_new_tokens=max_new_tokens,
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
        max_new_tokens=max_new_tokens,
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

    print(f"\nInput{i}: {entry['question']}")

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

for i, entry in enumerate(test_data[:5]):
    input_text = format_input(entry)

    # Reference Model Generation
    ref_input_ids = text_to_token_ids(input_text, tokenizer).to(device)
    ref_generated = generate(
        model=ref_model,
        idx=ref_input_ids.to(device),
        max_new_tokens=max_new_tokens,
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
        max_new_tokens=max_new_tokens,
        # temperature=temperature,
        # top_p=top_p,
        stopping_criteria=stopping_criteria,
        eot_token_id=eot_token_id
    )
    fine_tuned_model_full_text = fine_tuned_tokenizer.decode(fine_tuned_model_generated[0], skip_special_tokens=False)
    fine_tuned_model_response = postprocess_response(fine_tuned_model_full_text)

    print(f"\nInput{i}: {entry['question']}")
    print("\n ----- Reference Model ----- ")
    print(f"Reference Response: {ref_response}")

    print("\n ----- Policy Model ----- ")
    print(f"Policy Response: {fine_tuned_model_response}")

    print("\n ----- Expected Response ----- ")
    print(f"Expected Answer: {entry['chosen']}")
    print("="*80, "\n")

    with open(os.path.join(result_dir, "foutput_test.txt"), "a") as f:
        f.write(f"\nInput{i}: {entry['question']}")
        f.write("\n ----- Reference Model ----- ")
        f.write(f"Reference Response: {ref_response}")
        f.write("\n ----- Policy Model ----- ")
        f.write(f"Policy Response: {fine_tuned_model_response}")
        f.write("\n ----- Expected Response ----- ")
        f.write(f"Expected Answer: {entry['chosen']}")
        f.write("="*80 + "\n")
