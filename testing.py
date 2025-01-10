from utility import *
import config
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from functools import partial
import json
from typing import List, Tuple

model_name = config.model_name
token = config.token
device = torch.device("cpu")
print("Using CPU")
# Load a Hugging Face model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)#.to(device)
for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()
ref_model.to(device)

def load_model(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads a model and its tokenizer from the specified path.
    
    Args:
        model_path (str): Path or identifier of the model.
    
    Returns:
        Tuple containing the model and tokenizer.
    """
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # Set to True if using custom code
        device_map=device,       # Automatically map to available devices
        torch_dtype="auto"       # Automatically select dtype
    )
    # model.to(DEVICE)
    model.eval()
    return model, tokenizer

FINE_TUNED_MODEL_PATH = "Llama-3.2-1B-DPO"
fine_tuned_model, fine_tuned_tokenizer = load_model(FINE_TUNED_MODEL_PATH)
fine_tuned_model.eval()
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
            prompt = tokenizer.decode(tokenizer.encode(entry["instruction"], truncation=True, max_length=512))
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
for entry in val_data[:3]:

    input_text = format_input(entry)

    token_ids = generate(
        model=ref_model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    reference_response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    token_ids = generate(
        model=fine_tuned_model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    policy_response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nReference model response:\n>> {reference_response_text.strip()}")
    print(f"\nPolicy model response:\n>> {policy_response_text.strip()}")
    print("\n-------------------------------------\n")