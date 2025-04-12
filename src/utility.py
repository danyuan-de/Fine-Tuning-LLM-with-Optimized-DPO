import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer #,StoppingCriteria, StoppingCriteriaList, 
import math
import src.config as config

def _get_prefix(model: str, method: str, file: str, label: str = None) -> str:
    """Extract the file suffix based on a fixed mapping from the data file name."""
    model_short = model.split('/')[-1]
    training_dtype = next((dtype for dtype in ["content", "mixed", "structure", "preference"] if dtype in file), "unknown")
    if label is not None:
        return model_short + "_" + method.upper() + "_" + training_dtype + "_" + label
    return model_short + "_" + method.upper() + "_" + training_dtype

def _build_hyperparam_str(method: str, learning_rate: float = None, beta: float = None,
                          lambda_dpop: float = None, lambda_kl: float = None,
                          lambda_contrast: float = None) -> str:
    """Construct a hyperparameter string based on provided parameters."""
    parts = []
    if learning_rate is not None:
        parts.append(f"lr{learning_rate:.1e}")
    if beta is not None:
        parts.append(f"b{beta:.2f}")
    if lambda_dpop is not None and method in ['dpop', 'dpopkl']:
        parts.append(f"dp{lambda_dpop:.1f}")
    if lambda_kl is not None and method in ['dpokl', 'dpopkl']:
        parts.append(f"kl{lambda_kl:.2f}")
    if lambda_contrast is not None and method == 'dpocontrast':
        parts.append(f"c{lambda_contrast:.2f}")
    return "_".join(parts)

def get_output_filename(model: str, method: str, file: str, label: str = None, learning_rate: float = None,
                       beta: float = None, lambda_dpop: float = None, 
                       lambda_kl: float = None, lambda_contrast: float = None,
                       typename: str = "json") -> str:
    """
    Dynamically generate output filenames based on the method and data file.
    """
    prefix = _get_prefix(model, method, file, label)
    hyperparam_str = _build_hyperparam_str(method, learning_rate, beta, lambda_dpop, lambda_kl, lambda_contrast)

    if hyperparam_str:
        filename = f"{prefix}_{hyperparam_str}.{typename}"
    else:
        filename = f"{prefix}.{typename}"
    return os.path.join(config.result_dir, filename)

def get_dpo_params(method: str):
    """
    Returns a dictionary of relevant parameters for the specified DPO method.
    
    Args:
        method (str): The DPO method name ('dpo', 'dpop', 'dpokl', 'dpopkl', 'dpocontrast')
        config: Configuration object containing parameter values
        
    Returns:
        dict: Dictionary of parameters relevant to the specified method
    """
    # All methods require beta
    params = {'beta': config.beta}
    
    # Add method-specific parameters
    if method in ['dpop', 'dpopkl']:
        params['lambda_dpop'] = config.lambda_dpop
        
    if method in ['dpokl', 'dpopkl']:
        params['lambda_kl'] = config.lambda_kl
        
    if method == 'dpocontrast':
        params['lambda_contrast'] = config.lambda_contrast
    
    return params

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
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{input_text}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
        )
    
    elif "question" in entry:
        system_prompt = (
            "You are a physics expert assistant. "
            "Provide a detailed, reasoning process followed by a clear final answer for the following question."
        )
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"Question: {entry['question']}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
        )

# self-defined collate_fn for DataLoader
def custom_collate_fn(
    batch,
    eos_token_id=None,
    tokenizer=None,
    device=None,
    allowed_max_length=None,
    mask_prompt_tokens=None,
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
                torch.full((max_length_common - sequence_tensor.size(0),), 
                           fill_value=tokenizer.pad_token_id, 
                           dtype=torch.long).to(device)  # Move to device
            ])

            # Create a mask tensor to ignore the padding tokens
            mask = torch.ones_like(padded_sequence, dtype=torch.bool).to(device)  # Move to device
            mask[sequence_tensor.size(0):] = False  # Set padding tokens to False

            # Mask the prompt tokens if needed
            if mask_prompt_tokens:
                mask[:prompt.size(0)] = False

            # Make sure the EOS token is not masked
            if eos_token_id in sequence_tensor:
                eos_positions = (sequence_tensor == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    eos_pos = eos_positions[-1].item()  # Last EOS in response
                else:
                    eos_pos = sequence_tensor.size(0) - 1
            else:
                eos_pos = sequence_tensor.size(0) - 1

            mask[eos_pos] = True  # Set the EOS token to True

            # Ensure EOS token is unmasked
            eos_positions = (sequence_tensor == eos_token_id).nonzero(as_tuple=True)[0]
            eos_pos = eos_positions[-1].item() if len(eos_positions) > 0 else sequence_tensor.size(0) - 1
            mask[eos_pos] = True

            batch_data[key].append(padded_sequence)
            batch_data[f"{key}_mask"].append(mask)

    # Stack the tensors (already on device, no need for .to(device))
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        batch_data[key] = torch.stack(batch_data[key])

        # Truncate the sequences if needed
        if allowed_max_length is not None:
            batch_data[key] = batch_data[key][:, :allowed_max_length]

    return batch_data

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, label="loss", save_path=None):
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
    plt.savefig(save_path)
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
    # stopping_criteria=None,
    max_new_tokens=512,
    context_size=4096,
    temperature=0.0,
    top_k=None,
    top_p=None,  
    eos_token_id=None
):
    """
    Generates text given a starting sequence of token IDs.

    Args:
        model: The language model.
        idx (torch.LongTensor): The input token IDs, shape (batch_size, seq_len).
        stopping_criteria (StoppingCriteriaList): Any stopping criteria to be applied.
        max_new_tokens (int): The maximum number of tokens to generate.
        context_size (int): The maximum context size to preserve from the prompt.
        temperature (float): If > 0, controls the sampling randomness. 
                             If == 0, a pure greedy (argmax) decode is done.
        top_k (int): The number of highest probability tokens to keep for sampling.
        top_p (float): The cumulative probability threshold for top-p sampling.
        eos_token_id (int): The token ID marking end-of-text.

    Returns:
        torch.LongTensor: The token IDs of the generated text (including the prompt).
        
    enhanced generation function that supports both top-k and top-p
    priority: top_p > top_k (when both are set) 
    However, top_p and top_k are not used in this project, just for reference
    """

    device = model.device 
    idx = idx.to(device)

    for iteration in range(max_new_tokens):
        # Truncate the context
        idx_cond = idx[:, -context_size:]  # shape: (batch_size, <=context_size)
        
        # Compute logits; we only need the final token’s logits
        with torch.no_grad():
            output = model(idx_cond)
            logits = output.logits if hasattr(output, "logits") else output[0]
        logits = logits[:, -1, :]  # shape: (batch_size, vocab_size)

        # Decide how to pick the next token
        if temperature == 0.0:
            # Greedy decode: pick the argmax
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
        else:
            # Scale by temperature
            logits = logits / temperature

            # Apply top-p sampling (priority over top-k if both specified)
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Exclude tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift one token to keep the first token that crosses top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # Scatter back
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Apply top-k sampling (if top_p not used)
            elif top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))
                kth_logits, kth_indices = torch.topk(logits, top_k)
                # Value of the lowest logit in top-k
                min_val = kth_logits[:, -1].unsqueeze(-1)
                # Anything below that gets set to -inf
                logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(device), logits)

            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        # Concatenate the chosen token
        idx = torch.cat((idx, next_token), dim=1)

        # # Check if any stopping criteria are met; if so, exit the loop.
        # if stopping_criteria is not None:
        #     if any(sc(input_ids=idx, scores=logits) for sc in stopping_criteria):
        #         break

        # Check EOS or max tokens
        if eos_token_id is not None:
            if (next_token == eos_token_id).any():
                # If any in the batch hits EOS, you might choose to break or handle individually
                break

    return idx

# class EOSStoppingCriteria(StoppingCriteria):
#     def __init__(self, eos_token_id):
#         self.eos_token_id = eos_token_id
        
#     def __call__(self, input_ids, scores, **kwargs):
#         # check if the last token is the EOS token
#         return len(input_ids[0]) > 0 and input_ids[0][-1] == self.eos_token_id
    

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

def new_postprocess_response(full_text: str) -> str:
    """
    Process the response text from the model output.
    
    This function:
    1. Extracts assistant's response if model header tags are present
    2. Preserves instructional tags like <observation>, <think>, etc.
    3. Removes system/model tokens and special tokenizer tokens
    4. Handles tokenizer-specific tokens if a tokenizer is provided
    
    Args:
        full_text (str): The full text output from the model
        tokenizer: Optional tokenizer object to handle special tokens
        
    Returns:
        str: The processed response
    """
    # Extract the assistant's response if header tags are present
    if "<|start_header_id|>assistant<|end_header_id|>" in full_text:
        response = full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        
        # Handle EOT token if present
        if "<|eot_id|>" in response:
            response = response.split("<|eot_id|>")[0]
    else:
        # If no assistant header, use the full text
        response = full_text
    
    # List of system tokens to remove
    system_tokens = [
        "<|begin_of_text|>", 
        "<|start_header_id|>system<|end_header_id|>",
        "<|start_header_id|>user<|end_header_id|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|eot_id|>"
    ]
    
    # Remove system tokens
    for token in system_tokens:
        response = response.replace(token, "")
    
    # Remove any trailing URLs or web content (if needed)
    # response = response.replace("http://", "")
    # response = response.replace("https://", "")
    
    # Return the cleaned response
    return response.strip()

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