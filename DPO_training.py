# Description: This script trains a model using DPO on the instruction data with preferences.
# Execute: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python DPO_training.py
# Update pytorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/mps


import os
import config
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig
from functools import partial
from dpo_loss import DPOLoss
import copy
import time
from datetime import timedelta


# Replace this with a suitable model checkpoint from the Hugging Face Hub
model_name = config.model_name
token = config.token
os.environ['WANDB_API_KEY'] = config.WANDB_API_KEY


def get_device():
    """
    Determine the available device for computation.

    Returns:
        torch.device: The device to use (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

device = device = torch.device("cpu")
print("Using CPU")
# Load a Hugging Face model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


# Create a reference model for DPO by copying and freezing the parameters
ref_model = copy.deepcopy(model)
for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()

# Ensure pad_token is defined
if tokenizer.pad_token is None:
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        ref_model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = tokenizer.pad_token_id
ref_model.config.pad_token_id = tokenizer.pad_token_id

print("Model and tokenizer loaded.")

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

# Load your dataset
file_path = "instruction-data-with-preference.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

print("Number of entries:", len(data))

# Train/val/test split
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))


# collate_fn for DataLoader
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    allowed_max_length=None,
    mask_prompt_tokens=True,
    device="cpu"
):
    # Initialize lists to hold batch data
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [],
        "chosen_mask": []

    }

    # Determine the longest sequence to set a common padding length
    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            current_max = max(len(item[key])+1 for item in batch)
            max_length_common = max(max_length_common, current_max)

    # Process each item in the batch
    for item in batch:
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)

        for key in ["chosen", "rejected"]:
            # Adjust padding according to the common maximum length
            sequence = item[key]
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))
            mask = torch.ones(len(padded)).bool()

            # Set mask for all padding tokens to False
            mask[len(sequence):] = False

            # Set mask for all input tokens to False
            # +2 sets the 2 newline ("\n") tokens before "### Response" to False
            if mask_prompt_tokens:
                mask[:prompt.shape[0]+2] = False

            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    # Final processing
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        # Stack all sequences into a tensor for the given key
        tensor_stack = torch.stack(batch_data[key])

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]

        # Move to the specified device
        batch_data[key] = tensor_stack.to(device)

    return batch_data


# Prepare a custom PyTorch dataset
class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            prompt = format_input(entry)
            rejected_response = entry["rejected"]
            chosen_response = entry["chosen"]

            prompt_tokens = tokenizer.encode(prompt)
            chosen_full_text = f"{prompt}\n\n### Response:\n{chosen_response}"
            rejected_full_text = f"{prompt}\n\n### Response:\n{rejected_response}"
            chosen_full_tokens = tokenizer.encode(chosen_full_text)
            rejected_full_tokens = tokenizer.encode(rejected_full_text)

            self.encoded_texts.append({
                "prompt": prompt_tokens,
                "chosen": chosen_full_tokens,
                "rejected": rejected_full_tokens,
            })

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
    allowed_max_length=512   # The supported context length of the model
)

batch_size = 8
# Create datasets and dataloaders
train_dataset = PreferenceDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    collate_fn=customized_collate_fn, 
    drop_last=True, 
    shuffle=True)

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

# # DPO loss function
# def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta=0.1):
#     pi_logratios = chosen_seq_logp - rejected_seq_logp
#     ref_logratios = chosen_ref_seq_logp - rejected_ref_seq_logp
#     losses = -torch.nn.functional.logsigmoid(beta * (pi_logratios - ref_logratios))
#     return losses.mean()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Initialize the DPO loss function with beta=0.1
dpo_loss_fn = DPOLoss(beta=0.1)

ref_model.to(device)

# Example training loop
num_epochs = 3
start_training_time = time.time()
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}")
    epoch_training_start_time = time.time()
    model.train()
    total_loss = 0.0

    for index, batch in enumerate(train_loader, start=1):
        each_batch_start_time = time.time()
        prompt_batch = batch["prompt"]
        chosen_batch = batch["chosen"]
        rejected_batch = batch["rejected"]

        # Decode tensors back to text using the tokenizer
        prompt_texts = [tokenizer.decode(p, skip_special_tokens=True) for p in prompt_batch]
        chosen_texts = [tokenizer.decode(c, skip_special_tokens=True) for c in chosen_batch]
        rejected_texts = [tokenizer.decode(r, skip_special_tokens=True) for r in rejected_batch]

        # Re-encode the concatenated sequences for chosen and rejected responses
        chosen_inputs = tokenizer(
            [p + "\n\n### Response:\n" + c for p, c in zip(prompt_texts, chosen_texts)],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        chosen_inputs["labels"] = chosen_inputs["input_ids"].clone()
        chosen_inputs["labels"][chosen_inputs["input_ids"] == tokenizer.pad_token_id] = -100
        chosen_inputs = {k: v.to(device) for k, v in chosen_inputs.items()}

        rejected_inputs = tokenizer(
            [p + "\n\n### Response:\n" + r for p, r in zip(prompt_texts, rejected_texts)],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        rejected_inputs["labels"] = rejected_inputs["input_ids"].clone()
        rejected_inputs["labels"][rejected_inputs["input_ids"] == tokenizer.pad_token_id] = -100
        rejected_inputs = {k: v.to(device) for k, v in rejected_inputs.items()}

        # Compute logits for model
        chosen_outputs = model(**chosen_inputs)
        rejected_outputs = model(**rejected_inputs)

        # Compute logits for ref_model
        with torch.no_grad():
            chosen_ref_outputs = ref_model(**chosen_inputs)
            rejected_ref_outputs = ref_model(**rejected_inputs)

        # Get log probabilities
        chosen_log_probs = torch.nn.functional.log_softmax(chosen_outputs.logits, dim=-1)
        rejected_log_probs = torch.nn.functional.log_softmax(rejected_outputs.logits, dim=-1)
        chosen_ref_log_probs = torch.nn.functional.log_softmax(chosen_ref_outputs.logits, dim=-1)
        rejected_ref_log_probs = torch.nn.functional.log_softmax(rejected_ref_outputs.logits, dim=-1)

        chosen_labels = chosen_inputs["labels"]
        rejected_labels = rejected_inputs["labels"]

        # Replace -100 with pad_token_id before gather
        chosen_labels_safe = chosen_labels.clone()
        chosen_labels_safe[chosen_labels_safe == -100] = tokenizer.pad_token_id

        rejected_labels_safe = rejected_labels.clone()
        rejected_labels_safe[rejected_labels_safe == -100] = tokenizer.pad_token_id

        # Gather log probs at label positions using the safe labels
        chosen_token_logps = chosen_log_probs.gather(-1, chosen_labels_safe.unsqueeze(-1)).squeeze(-1)
        rejected_token_logps = rejected_log_probs.gather(-1, rejected_labels_safe.unsqueeze(-1)).squeeze(-1)
        chosen_ref_token_logps = chosen_ref_log_probs.gather(-1, chosen_labels_safe.unsqueeze(-1)).squeeze(-1)
        rejected_ref_token_logps = rejected_ref_log_probs.gather(-1, rejected_labels_safe.unsqueeze(-1)).squeeze(-1)

        chosen_mask = chosen_labels != -100
        rejected_mask = rejected_labels != -100

        # Compute average log probabilities per sequence
        chosen_seq_logp = (chosen_token_logps * chosen_mask).sum(dim=1) / chosen_mask.sum(dim=1)
        rejected_seq_logp = (rejected_token_logps * rejected_mask).sum(dim=1) / rejected_mask.sum(dim=1)
        chosen_ref_seq_logp = (chosen_ref_token_logps * chosen_mask).sum(dim=1) / chosen_mask.sum(dim=1)
        rejected_ref_seq_logp = (rejected_ref_token_logps * rejected_mask).sum(dim=1) / rejected_mask.sum(dim=1)

        # Compute DPO loss using the DPOLoss class
        loss = dpo_loss_fn(
            pi_logp_chosen=chosen_seq_logp,
            pi_logp_rejected=rejected_seq_logp,
            ref_logp_chosen=chosen_ref_seq_logp,
            ref_logp_rejected=rejected_ref_seq_logp,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        each_batch_end_time = time.time()
        total_time_in_batch = each_batch_end_time - each_batch_start_time
        print(f"Batch {index} took {str(timedelta(seconds=total_time_in_batch))}")

    avg_loss = total_loss / len(train_loader)
    epoch_training_end_time = time.time()
    total_time_in_epoch = epoch_training_end_time - epoch_training_start_time
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f} took {str(timedelta(seconds=total_time_in_epoch))}")

end_training_time = time.time()
print("Training took", str(timedelta(seconds=end_training_time - start_training_time)))

save_path = "./Llama-3.2-1B-DPO"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved to {save_path}")