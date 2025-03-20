import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import re
from transformers import StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM, AutoTokenizer
import math
import src.config as config

# Get the device to use
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

def format_input(entry):
    if "instruction" in entry:
        system_prompt = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        ) # for the instruction-data-with-preference.json
        input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else "" # for the instruction-data-with-preference.json
        return system_prompt + input_text
    
    elif "question" in entry:
        system_prompt = (
            "You are a physics expert assistant. "
            "Provide a detailed, reasoning process followed by a clear final answer for the following question."
        )
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"Question: {entry['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

# self-defined collate_fn for DataLoader
def custom_collate_fn(
    batch,
    eot_token_id=None,
    tokenizer=None,
    device=None,
    allowed_max_length=None,
    mask_prompt_tokens=None,
):
    """
    A modified collate function for DPO training with Llama-3.1-8B.
    
    Args:
        batch: List of dictionaries containing prompt, chosen, and rejected sequences
        eot_token_id: Token ID for end-of-text
        tokenizer: The tokenizer used
        device: Device to put tensors on
        allowed_max_length: Maximum sequence length to allow
        mask_prompt_tokens: Whether to mask prompt tokens in loss calculation
    
    Returns:
        Dictionary with batched tensors for training
    """
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
            current_max = max(len(item[key])+1 for item in batch)
            max_length_common = max(max_length_common, current_max)
    
    # Ensure we don't exceed the model's context window
    if allowed_max_length is not None:
        max_length_common = min(max_length_common, allowed_max_length)

    # Process each item in the batch
    for item in batch:
        prompt = torch.tensor(item["prompt"], dtype=torch.long)
        batch_data["prompt"].append(prompt)

        for key in ["chosen", "rejected"]:
            sequence = torch.tensor(item[key], dtype=torch.long)
            sequence_tensor = sequence.to(device) if device else sequence
            
            # Prepare padding tensor if needed
            pad_length = max_length_common - len(sequence_tensor)
            if pad_length > 0:
                padding = torch.full((pad_length,), tokenizer.pad_token_id if tokenizer else eot_token_id, 
                                    dtype=torch.long, device=device)
                padded_sequence = torch.cat([sequence_tensor, padding])
            else:
                padded_sequence = sequence_tensor
            
            # Create mask tensor (True for tokens to include in loss, False for others)
            mask = torch.zeros_like(padded_sequence, dtype=torch.bool, device=device)
            
            # Set sequence tokens to True (excluding padding)
            mask[:len(sequence_tensor)] = True
            
            # Mask prompt tokens if requested
            if mask_prompt_tokens and len(prompt) < len(sequence_tensor):
                mask[:len(prompt)+2] = False  # +2 to account for newline tokens
            
            # Ensure EOT token is correctly handled
            if eot_token_id is not None and eot_token_id in sequence_tensor:
                eot_positions = (sequence_tensor == eot_token_id).nonzero(as_tuple=True)[0]
                if len(eot_positions) > 0:
                    eot_pos = eot_positions[-1].item()
                    mask[eot_pos] = True
            
            batch_data[key].append(padded_sequence)
            batch_data[f"{key}_mask"].append(mask)

    # Stack tensors (they're already on device)
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        batch_data[key] = torch.stack(batch_data[key])
        
        # Truncate if needed
        if allowed_max_length is not None:
            batch_data[key] = batch_data[key][:, :allowed_max_length]

    return batch_data

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label=f"Training {label}")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(os.path.join(config.result_dir, f"{label}-plot.png"))
    # plt.show()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text) # , allowed_special={"<|endoftext|>"}) --> it's OpenAI's tiktoken
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate(
    model, 
    idx, 
    stopping_criteria=None,
    max_new_tokens=None,
    context_size=4096, 
    temperature=config.temperature,
    top_k=None,
    top_p=None,  
    eot_token_id=None
):
    """
    Args:
        idx: the input token IDs
        stopping_criteria: a function that determines when to stop the generation process
        max_new_tokens: the maximum number of tokens to generate
        context_size: the maximum number of tokens that are considered as the contextual input 
                    when generating the next token
        temperature: the temperature value for sampling
        top_k: the number of highest probability tokens to keep for sampling
        top_p: the cumulative probability threshold for top-p sampling
        eot_token_id: the token ID for the end-of-text token
    Returns:
        idx: the token IDs of the generated text
        
    enhanced generation function that supports both top-k and top-p
    priority: top_p > top_k (when both are set)
    """

    device = model.device 
    idx = idx.to(device)

    # Limit the maximum number of iterations to prevent infinite generation (iteration)
    for iteration in range(max_new_tokens):
        # truncate context
        idx_cond = idx[:, -context_size:]#.to(device) 
        
        # get logits
        with torch.no_grad():
            output = model(idx_cond)
            logits = output.logits if hasattr(output, "logits") else output[0]
        
        # select the last token's logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        
        # Check top_p and top_k are valid
        if top_p is not None:
            assert 0 < top_p <= 1.0, f"top_p must be between 0 and 1, got {top_p}"
        if top_k is not None:
            assert top_k > 0, f"top_k must be positive, got {top_k}"
            
        # prioritize top-p sampling
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), 
                dim=-1
            )
            
            # remove tokens with cumulative probability above top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # keep the first token above top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            # create mask
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # secondary top-k sampling
        elif top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, float('-inf'), logits)

        # Probability Calculation and Sampling
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(device)

        # EOT check
        if (eot_token_id is not None and idx_next.item() == eot_token_id) or iteration == max_new_tokens - 1:
                break

        idx = torch.cat((idx.to(device), idx_next), dim=1)

    return idx

class EOTStoppingCriteria(StoppingCriteria):
    def __init__(self, eot_token_id):
        self.eot_token_id = eot_token_id
        
    def __call__(self, input_ids, scores, **kwargs):
        # check if the last token is the EOT token
        return len(input_ids[0]) > 0 and input_ids[0][-1] == self.eot_token_id
    

# postprocess response to remove unwanted tokens
def postprocess_response(full_text: str) -> str:
    # make sure only the first assistant part is kept
    if "<|start_header_id|>assistant<|end_header_id|>" in full_text:
        response = full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        # remove all subsequent HTML tags
        response = re.sub(r"<[^>]+>", "", response)
        # Remove any trailing non-physics content (e.g., URLs, Bootstrap)
        response = re.sub(r"://.*$", "", response, flags=re.DOTALL)
        # cutoff at the first EOT token
        if "<|eot_id|>" in response:
            response = response.split("<|eot_id|>")[0]
        return response.strip()
    return "Invalid response format"

def calculate_perplexity(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: str) -> float:
    """
    Calculates the perplexity of the model on the given text.
    
    Args:
        model (AutoModelForCausalLM): The language model.
        tokenizer (AutoTokenizer): The tokenizer.
        text (str): The text to evaluate.
    
    Returns:
        Perplexity score as a float.
    """
    device = model.device
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[:, :512].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    perplexity = math.exp(loss.item())
    return perplexity