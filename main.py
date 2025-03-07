# Description: This script trains a model using DPO on the instruction data with preferences.
# Execute: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python DPO_training.py for MPS (MacOS)
# Execute: python DPO_training.py for CUDA (Linux)
# Update pytorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/mps


import os
import config
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig
from functools import partial
from dpoLoss import DPOLoss
import copy
import time
from datetime import timedelta
from utility import *
import multiprocessing

# --------- File Paths ---------
model_name = config.model_name
cache_dir = config.cache_dir
file_path = config.file_content

# --------- Hyperparameters ---------
allowed_max_length = config.allowed_max_length
max_new_tokens = config.max_new_tokens
batch_size = config.batch_size
num_epochs = config.num_epochs
beta = config.beta
learning_rate = config.learning_rate
temperature = config.temperature
top_p = config.top_p
dpo_loss_fn = DPOLoss(beta=beta)

# --------- Device ---------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

# --------- Load a Hugging Face model and tokenizer ---------
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16)

# Add the EOT token to the tokenizer
eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

special_tokens = {
    "additional_special_tokens": ["<|eot_id|>"]
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer)) # adjust the size of the token embeddings

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
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        ref_model.resize_token_embeddings(len(tokenizer))

# model.config.pad_token_id = tokenizer.pad_token_id
# ref_model.config.pad_token_id = tokenizer.pad_token_id

print("Model and tokenizer loaded.")

# Load the data
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

print("Number of entries:", len(data))

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


# collate_fn for DataLoader
def custom_collate_fn(
    batch,
    eot_token_id=eot_token_id,
    allowed_max_length=None,
    mask_prompt_tokens=True,
    device=device
):
    # Initialize the batch data
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "chosen_mask": [],
        "rejected_mask": []
    }

    # Calculate the maximum length of the chosen and rejected sequences
    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            current_max = max(len(item[key]) for item in batch)
            max_length_common = max(max_length_common, current_max)

    # Process each item in the batch
    for item in batch:
        prompt = torch.tensor(item["prompt"], dtype=torch.long).to(device)
        batch_data["prompt"].append(prompt)

        for key in ["chosen", "rejected"]:
            sequence = item[key]
            sequence_tensor = torch.tensor(sequence, dtype=torch.long).to(device)  # Move to device immediately

            # Fill the sequence tensor with padding tokens to match the maximum length
            padded_sequence = torch.cat([
                sequence_tensor,
                torch.full((max_length_common - sequence_tensor.size(0),), fill_value=tokenizer.pad_token_id, dtype=torch.long).to(device)  # Move to device
            ])  # Change fill_value=eot_token_id

            # Create a mask tensor to ignore the padding tokens
            mask = torch.ones_like(padded_sequence, dtype=torch.bool).to(device)  # Move to device
            mask[sequence_tensor.size(0):] = False  # Set padding tokens to False

            # Mask the prompt tokens if needed
            if mask_prompt_tokens:
                mask[:prompt.size(0)] = False

            # Make sure the EOT token is not masked
            if eot_token_id in sequence_tensor:
                eot_positions = (sequence_tensor == eot_token_id).nonzero(as_tuple=True)[0]
                if len(eot_positions) > 0:
                    eot_pos = eot_positions[-1].item()  # Last EOT in response
                else:
                    eot_pos = sequence_tensor.size(0) - 1
            else:
                eot_pos = sequence_tensor.size(0) - 1

            mask[eot_pos] = True  # Set the EOT token to True

            # Ensure EOT token is unmasked
            eot_positions = (sequence_tensor == eot_token_id).nonzero(as_tuple=True)[0]
            eot_pos = eot_positions[-1].item() if len(eot_positions) > 0 else sequence_tensor.size(0) - 1
            mask[eot_pos] = True

            batch_data[key].append(padded_sequence)
            batch_data[f"{key}_mask"].append(mask)

    # Stack the tensors (already on device, no need for .to(device))
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        batch_data[key] = torch.stack(batch_data[key])

        # Truncate the sequences if needed
        if allowed_max_length is not None:
            batch_data[key] = batch_data[key][:, :allowed_max_length]

    return batch_data


# Prepare a custom PyTorch dataset
class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.eot_token = "<|eot_id|>"  # Define EOT token for Llama 3 models

        # Verify EOT token exists in tokenizer
        if self.eot_token not in tokenizer.added_tokens_decoder:
            tokenizer.add_special_tokens({"additional_special_tokens": [self.eot_token]})

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            prompt = format_input(entry)
            
            # Add EOT token to responses
            chosen_response = entry["chosen"] + self.eot_token
            rejected_response = entry["rejected"] + self.eot_token

            # Tokenize full texts
            chosen_full_text = f"{prompt}\n{chosen_response}"
            rejected_full_text = f"{prompt}\n{rejected_response}"

            # tokenize the full texts
            chosen_full_tokens = self.tokenizer.encode(chosen_full_text)
            rejected_full_tokens = self.tokenizer.encode(rejected_full_text)

            self.encoded_texts.append({
                "prompt": self.tokenizer.encode(prompt),
                "chosen": chosen_full_tokens,
                "rejected": rejected_full_tokens,
            })
        print("\n=== data pre-processing validation ===")
        sample_entry = self.encoded_texts[0]
        print("Prompt:", tokenizer.decode(sample_entry["prompt"]))
        print("Chosen:", tokenizer.decode(sample_entry["chosen"]))
        print("Rejected:", tokenizer.decode(sample_entry["rejected"]))

    def __getitem__(self, index):
        item = self.encoded_texts[index]
        print(f"Index: {index}, Prompt size: {len(item['prompt'])}, Chosen size: {len(item['chosen'])}, Rejected size: {len(item['rejected'])}")
        return item

    def __len__(self):
        return len(self.data)

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,            # Put the data directly on a GPU if available
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


optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader), eta_min=1e-6)

# Before training loop. If chosen and rejected responses are too similar, the preference margin won’t grow.
batch = next(iter(train_loader))
print("Chosen sample:", tokenizer.decode(batch["chosen"][0].tolist()))
print("Rejected sample:", tokenizer.decode(batch["rejected"][0].tolist()))

# Define the training function
def train_model_dpo_simple(
    policy_model, reference_model, train_loader, val_loader,
    optimizer, num_epochs, beta,
    eval_freq, eval_iter):
    """
    Fine-tunes the policy model using the DPO method.

    :param policy_model: The model to be fine-tuned.
    :param reference_model: The reference model (with frozen weights).
    :param train_loader: The DataLoader for the training dataset.
    :param val_loader: The DataLoader for the validation dataset.
    :param optimizer: The optimizer.
    :param num_epochs: The number of training epochs.
    :param beta: The beta value used in the DPO loss.
    :param eval_freq: The frequency (in steps) at which to perform evaluations.
    :param eval_iter: The number of evaluation iterations.
    :return: A dictionary tracking various losses and reward metrics.
    """
    print("Starting training...")
    print("Ref model grad status:", next(ref_model.parameters()).requires_grad)
    # Initialize lists to track losses and tokens seen
    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": []
    }
    tokens_seen, global_step = 0, -1

    sample_entry = val_data[0] if val_data else None # Sample entry for generation

    # Main training loop
    for epoch in range(num_epochs):
        policy_model.train()  # Set model to training mode

        for batch_idx, batch in enumerate(train_loader):

            optimizer.zero_grad()
            # with autocast():  # Enable mixed precision
            loss, chosen_rewards, rejected_rewards = dpo_loss_fn.compute_dpo_loss_batch(
                    batch=batch,
                    policy_model=policy_model,
                    reference_model=reference_model
            )
            # print(f"Step {global_step+1}: Loss before backward: {loss.item():.4f}")
            loss.backward()  # Direct backward pass without scaling
            grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)  # Optional clipping
            # print(f"Step {global_step+1}: Grad norm: {grad_norm.item():.4f}")
            param_before = next(policy_model.parameters()).clone().sum().item()
            optimizer.step() # Direct optimizer step
            param_after = next(policy_model.parameters()).sum().item()
            # print(f"Step {global_step+1}: Param sum change: {param_after - param_before:.6f}")
            scheduler.step()  # Update learning rate after optimizer step

            # tokens_seen = torch.tensor(0, dtype=torch.int64) # avoid overflow by using torch.tensor with dtype int64
            tokens_seen += batch["chosen"].numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                res = dpo_loss_fn.evaluate_dpo_loss_loader(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    eval_iter=eval_iter
                )
                # if sample_entry and (global_step // eval_freq) % 2 == 0:  # generate every 2nd evaluation
                #     policy_model.eval()
                #     with torch.no_grad():
                #         try:
                #             # prepare input
                #             input_text = format_input(sample_entry)
                #             token_ids = text_to_token_ids(input_text, tokenizer).to(device)
                            
                #             # generation config
                #             generation_config = {
                #                 'max_new_tokens': max_new_tokens,
                #                 'temperature': temperature,
                #                 'top_p': top_p,
                #                 'eot_token_id': eot_token_id
                #             }
                            
                #             # execute generation
                #             generated = generate(
                #                 model=policy_model,
                #                 idx=token_ids, #.to(device),
                #                 stopping_criteria=stopping_criteria,
                #                 **generation_config
                #             )
                            
                #             # post-process the generated text
                #             full_text = token_ids_to_text(generated, tokenizer)
                #             response = full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                #             response = response.split("<|eot_id|>")[0].strip()

                #         except Exception as e:
                #             response = f"~~~ Generation Error: {str(e)}"
                #             print(f"Generation failed at step {global_step}: {str(e)}")

                #         finally:
                #             policy_model.train()

                #     # Print the generated response
                #     print(f"\n{'='*40} Generation Sample (Step {global_step}) {'='*40}")
                #     print(f"[Input]\n{sample_entry['question']}")
                #     print(f"\n[Generated Response]\n{response}")
                #     print(f"[Expected Response]\n{sample_entry['chosen']}")
                #     print('='*90 + '\n')

                tracking["train_losses"].append(res["train_loss"])
                tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                tracking["tokens_seen"].append(tokens_seen)
                train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                    f"Train reward margins {train_reward_margin:.3f}, "
                    f"Val reward margins {val_reward_margin:.3f}"
                )
    print("Training completed.")
    return tracking

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
    policy_model=policy_model,
    reference_model=ref_model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    num_epochs=num_epochs,
    beta=beta, 
    eval_freq=5,
    eval_iter=5,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes (in {str(timedelta(seconds=end_time - start_time))})")

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
    train_loader=val_loader,
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
        stopping_criteria=stopping_criteria
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
        stopping_criteria=stopping_criteria
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
    train_loader=test_loader,
    val_loader=test_loader,
    eval_iter=5
)

print("Test loss:", test_res["val_loss"])
print("Test reward margin:", test_res["val_chosen_reward"] - test_res["val_rejected_reward"])

for i, entry in enumerate(test_data[:train_portion//2]):
    input_text = format_input(entry)

    # Reference Model Generation
    ref_input_ids = text_to_token_ids(input_text, tokenizer).to(device)
    ref_generated = generate(
        model=ref_model,
        idx=ref_input_ids.to(device),
        max_new_tokens=max_new_tokens,
        # temperature=temperature,
        # top_p=top_p,
        stopping_criteria=stopping_criteria
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
        stopping_criteria=stopping_criteria
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

    with open("result.txt", "a") as f:
        f.write(f"\nInput{i}: {entry['question']}")
        f.write("\n ----- Reference Model ----- ")
        f.write(f"Reference Response: {ref_response}")
        f.write("\n ----- Policy Model ----- ")
        f.write(f"Policy Response: {fine_tuned_model_response}")
        f.write("\n ----- Expected Response ----- ")
        f.write(f"Expected Answer: {entry['chosen']}")
        f.write("="*80 + "\n")
