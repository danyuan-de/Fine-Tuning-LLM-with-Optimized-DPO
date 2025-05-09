import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import csv
from tqdm import tqdm
from typing import Dict, Union, List
import re
import src.config as config


# ------------------------------- File Management -------------------------------
def _get_prefix(method: str, file: str, model: str = None, label: str = None) -> str:
    """Extract the file suffix based on a fixed mapping from the data file name."""
    parts = []
    if model:
        parts.append(model.split('/')[-1])
    parts.append(method.upper())
    training_dtype = next((dtype for dtype in ["content", "mixed", "html", "chat", "structure", "preference", "data"] if dtype in file), "unknown")
    parts.append(training_dtype)
    if label:
        parts.append(label)
    return "_".join(parts)


# ------------------------------- Hyperparameter Management -------------------------------
def _build_hyperparam_str(method: str, learning_rate: float = None, beta: float = None,
                          lambda_dpop: float = None, lambda_shift: float = None) -> str:
    """Construct a hyperparameter string based on provided parameters."""
    parts = []
    if learning_rate is not None:
        parts.append(f"lr{learning_rate:.1e}")
    if beta is not None:
        parts.append(f"b{beta:.2f}")
    if lambda_dpop is not None and method in ['dpop', 'dpopshift']:
        parts.append(f"dp{lambda_dpop:.1f}")
    if lambda_shift is not None and method in ['dposhift', 'dpopshift']:
        parts.append(f"shift{lambda_shift:.2f}")
    return "_".join(parts)


# ------------------------------- Output Filename Management -------------------------------
def get_output_filename(method: str, file: str, model: str = None, label: str = None, learning_rate: float = None,
                        beta: float = None, lambda_dpop: float = None,
                        lambda_shift: float = None, avg: bool = config.average_log_probs, typename: str = "json") -> str:
    """
    Dynamically generate output filenames based on the method and data file.
    """
    prefix = _get_prefix(method, file, model, label)
    hyperparam_str = _build_hyperparam_str(method, learning_rate, beta, lambda_dpop, lambda_shift)
    mode = "avg" if avg else "sum"

    filename = f"{mode}_{prefix}"
    if hyperparam_str:
        filename += f"_{hyperparam_str}"
    filename += f".{typename}"

    return os.path.join(config.result_dir, filename)


# ------------------------------- DPO Parameters Management -------------------------------
def get_dpo_params(method: str):
    """
    Returns a dictionary of relevant parameters for the specified DPO method.

    Args:
        method (str): The DPO method name ('dpo', 'dpop', 'dposhift', 'dpopshift')
        config: Configuration object containing parameter values

    Returns:
        dict: Dictionary of parameters relevant to the specified method
    """
    # All methods require beta
    params = {'beta': config.beta}

    # Add method-specific parameters
    if method in ['dpop', 'dpopshift']:
        params['lambda_dpop'] = config.lambda_dpop

    if method in ['dposhift', 'dpopshift']:
        params['lambda_shift'] = config.lambda_shift

    return params


# ------------------------------- Logging Management -------------------------------
def log_result_csv(
    filename: str,
    **kwargs
):
    # Mapping of method to headers
    headers = [
        "epoch_frac",
        "step",
        "train_loss",
        "val_loss",
        "train_reward_margin",
        "val_reward_margin",
        "train_reward_accuracy",
        "val_reward_accuracy",
        "policy_chosen_logprobs",
        "policy_rejected_logprobs",
        "reference_chosen_logprobs",
        "reference_rejected_logprobs"
    ]

    row = [kwargs.get(h, None) for h in headers]

    os.makedirs(config.result_dir, exist_ok=True)
    write_header = not os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerow(row)

    print(f"✔️ Results logged to {filename}")


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
    # constants for clarity
    HEADER_SYSTEM = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    HEADER_USER = "<|start_header_id|>user<|end_header_id|>\n"
    HEADER_ASSISTANT = "<|start_header_id|>assistant<|end_header_id|>\n"
    EOT = "<|eot_id|>"

    # ── Orca DPO format ──
    if entry.get("system") is not None and entry.get("question") is not None:
        prompt = entry["system"].strip()
        question = entry.get("question").strip()
        return (
            f"{HEADER_SYSTEM}"
            f"{prompt}{EOT}"
            f"{HEADER_USER}"
            f"Question: {question}{EOT}"
            f"{HEADER_ASSISTANT}"
        )

    # instruction-style examples
    if entry.get("instruction"):
        preference_system_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{entry['instruction']}"
        )
        # optional extra input field
        input_part = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
        return (
            f"{HEADER_SYSTEM}"
            f"{preference_system_prompt}{EOT}"
            f"{HEADER_USER}"
            f"{input_part}{EOT}"
            f"{HEADER_ASSISTANT}"
        )

    # QA-style (either 'question' or fallback to 'content')
    q = entry.get("question") or entry.get("content")
    if entry.get("options"):
        if not config.MMLU_PRO_category_isPhysics:
            system_prompt = config.system_prompt_mc_general
        else:
            system_prompt = config.system_prompt_mc_physics
        opts = entry["options"]
        choices = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))
        return (
            f"{HEADER_SYSTEM}"
            f"{system_prompt}{EOT}"
            f"{HEADER_USER}"
            f"Question: {q}{EOT}"
            f"Options:\n{choices}{EOT}"
            f"{HEADER_ASSISTANT}"
        )
    else:
        system_prompt = config.system_prompt_physics
        return (
            f"{HEADER_SYSTEM}"
            f"{system_prompt}{EOT}"
            f"{HEADER_USER}"
            f"Question: {q}{EOT}"
            f"{HEADER_ASSISTANT}"
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

            # # Ensure EOS token is unmasked
            # eos_positions = (sequence_tensor == eos_token_id).nonzero(as_tuple=True)[0]
            # eos_pos = eos_positions[-1].item() if len(eos_positions) > 0 else sequence_tensor.size(0) - 1
            # mask[eos_pos] = True

            batch_data[key].append(padded_sequence)
            batch_data[f"{key}_mask"].append(mask)

    batch_data["prompt"] = pad_sequence(
        batch_data["prompt"],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    # Stack the tensors (already on device, no need for .to(device))
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        batch_data[key] = torch.stack(batch_data[key])

        # Truncate the sequences if needed
        if allowed_max_length is not None:
            batch_data[key] = batch_data[key][:, :allowed_max_length]
    
    batch_data["question_texts"] = [inst["question_text"] for inst in batch]
    batch_data["chosen_texts"]   = [inst["chosen_text"]   for inst in batch]
    batch_data["rejected_texts"] = [inst["rejected_text"] for inst in batch]


    return batch_data


# def custom_collate_fn(
#     batch, tokenizer, device, allowed_max_length=None, mask_prompt=False
# ):
#     fields = ["prompt", "chosen", "rejected"]
#     data = {k: [] for k in fields}
#     eos_id = tokenizer.eos_token_id
#     for inst in batch:
#         for k in fields:
#             data[k].append(torch.tensor(inst[k], dtype=torch.long))

#     # pad + stack
#     for k in fields:
#         padded = pad_sequence(
#             data[k], batch_first=True, padding_value=tokenizer.pad_token_id
#         )
#         if allowed_max_length:
#             padded = padded[:, :allowed_max_length]
#         data[k] = padded.to(device)

#     # attention masks
#     data["prompt_mask"] = (data["prompt"] != tokenizer.pad_token_id).long()
#     data["chosen_mask"] = (data["chosen"] != tokenizer.pad_token_id).long()
#     data["rejected_mask"] = (data["rejected"] != tokenizer.pad_token_id).long()

#     # mask the prompt tokens
#     if mask_prompt:
#         seq_len = data["prompt"].size(1)
#         data["chosen_mask"][:, :seq_len] = 0
#         data["rejected_mask"][:, :seq_len] = 0
#         # chosen
#         eos_pos = (data["chosen"] == eos_id).long().argmax(dim=1)
#         data["chosen_mask"][torch.arange(eos_pos.size(0)), eos_pos] = 1
#         # rejected
#         eos_pos = (data["rejected"] == eos_id).long().argmax(dim=1)
#         data["rejected_mask"][torch.arange(eos_pos.size(0)), eos_pos] = 1
    
#     data["question_texts"] = [inst["question_text"] for inst in batch]
#     data["chosen_texts"]   = [inst["chosen_text"]   for inst in batch]
#     data["rejected_texts"] = [inst["rejected_text"] for inst in batch]

#     return data


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


def text_to_token_ids(texts: Union[str, List[str]], tokenizer) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]

    encoding = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.allowed_max_length
    )
    return encoding.input_ids  # shape = (bsz, seq_len)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer, skip_special_tokens: bool = False) -> List[str]:
    all_ids = token_ids.tolist()
    texts = tokenizer.batch_decode(all_ids, skip_special_tokens=skip_special_tokens)
    # remove empty spaces
    return [t.strip() for t in texts]


def generate(
    model,
    idx,
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

        # Check EOS or max tokens
        if eos_token_id is not None:
            if (next_token == eos_token_id).any():
                # If any in the batch hits EOS, you might choose to break or handle individually
                break

    return idx


# ---------------------------- Postprocessing Model Output ----------------------------
def postprocess_response(full_text: str) -> str:
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
    
    # Remove any reserved_special_token_N placeholders
    response = re.sub(r"<\|reserved_special_token_\d+\|>", "", response)

    # Return the cleaned response
    return response.strip()


# ------------------------------- Perplexity Calculation -------------------------------
def calculate_perplexity(
    model: AutoModelForCausalLM,  # Language model for computing perplexity
    tokenizer: AutoTokenizer,     # Tokenizer for encoding input texts
    texts: str | list[str],       # Input text(s) to evaluate
    max_length: int = None,       # Maximum sequence length for truncation
    stride: int = None,           # Stride for splitting long sequences
    batch_size: int = 1,          # Number of texts to process in a batch
    device: torch.device = None   # Device to run the model on
) -> float | list[float]:         # Returns perplexity score(s)
    """
    Calculates the perplexity of the model on the given text(s).

    Args:
        model (AutoModelForCausalLM): The language model.
        tokenizer (AutoTokenizer): The tokenizer.
        texts (str or list[str]): The text(s) to evaluate.
        max_length (int, optional): Maximum sequence length for truncation. Defaults to model.config.max_position_embeddings.
        stride (int, optional): Stride for splitting long sequences. If None, no splitting is performed.
        batch_size (int, optional): Number of texts to process in a batch. Defaults to 1.
        device (torch.device, optional): Device to run the model on. Defaults to model's device.

    Returns:
        float or list[float]: Perplexity score(s) for the input text(s). Returns float("inf") for invalid inputs.
    """
    # Convert single string to list for uniform processing
    if isinstance(texts, str):
        texts = [texts]

    # Check for empty or invalid inputs
    if not texts or any(not text.strip() for text in texts):
        return [float("inf")] * len(texts) if len(texts) > 1 else float("inf")

    # Use model's device if none specified
    device = device or model.device

    # Set max_length to minimum of config or model limit
    max_length = max_length or min(config.allowed_max_length, model.config.max_position_embeddings)

    # Initialize list to store perplexity scores
    perplexities = []
    
    model.to(device).eval()  # Set model to evaluation mode
    pad_token_id = tokenizer.pad_token_id  # Get pad token ID

    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        # Extract current batch of texts
        batch_texts = texts[i:i + batch_size]

        try:
            # Tokenize texts with padding and truncation
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",            # Return PyTorch tensors
                truncation=(stride is None),    # Truncate to max_length
                max_length=max_length,          # Maximum sequence length
                padding=True,                   # Pad sequences to equal length
                add_special_tokens=True         # Add special tokens (e.g., BOS, EOS)
            )
            # Move input tensors to specified device
            input_ids = encodings.input_ids.to(device)
            attention_mask = encodings.attention_mask.to(device)

            # prepare labels with ignore_index
            labels = input_ids.clone()
            labels[input_ids == pad_token_id] = -100  # Set labels to -100 for padding tokens

            bsz, seq_len = input_ids.shape  
            # Handle long sequences with striding
            if stride and seq_len > max_length:
                print("==== Running with striding... ====")
                print(f"Sequence length {seq_len} exceeds max_length {max_length}. Using striding.")
                
                # total NLL and total token count per example
                total_nll = torch.zeros(bsz, device=device)
                total_tokens = torch.zeros(bsz, device=device)

                # Iterate over sequence with stride
                for start in range(0, seq_len, stride):
                    end = min(start + max_length, seq_len)

                    with torch.no_grad():
                        outputs = model(
                            input_ids=input_ids[:, start:end],
                            attention_mask=attention_mask[:, start:end],
                            labels=labels[:, start:end]
                        )
                    # outputs.loss is averaged over non-ignored tokens of this chunk
                    # Get chunk_loss * token_count_chunk and accumulate
                    # chunk_nll = loss * N_chunk_nonpad
                    # N_chunk_nonpad = chunk_mask.sum(dim=1)
                    token_count = attention_mask[:, start:end].sum(dim=1)          # shape (bsz,)
                    # expand scalar to per-example tensor
                    total_nll += outputs.loss * token_count
                    total_tokens += token_count

                    if end == seq_len: break

                # Final average Negative Log Likelihood (NLL)
                mean_nll = total_nll / total_tokens
                perplexities.extend(torch.exp(mean_nll).tolist())

            else:
                print("==== Running without striding... ====")
                # No striding, just compute loss for the entire batch
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                # outputs.loss -- averaged over non-ignored tokens
                ppl_batch = math.exp(outputs.loss.item())
                # Same perplexity for all samples in the batch
                perplexities.extend([ppl_batch] * bsz)

        except Exception as e:
            # Handle errors during batch processing
            print(f"Error processing batch {i // batch_size}: {e}")
            perplexities.extend([float("inf")] * len(batch_texts))

    # Return single value or list based on input
    return perplexities if len(perplexities) > 1 else perplexities[0]


# ------------------------------- Logging Perplexity to CSV -------------------------------
def log_ppl_csv(filename: str, *, model_name: str,
                chosen_ppl: float, rejected_ppl: float, self_ppl: float):
    headers = ["model", "chosen_ppl", "rejected_ppl", "self_ppl"]
    write_header = not os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(headers)
        w.writerow([model_name, chosen_ppl, rejected_ppl, self_ppl])


def summarize_ppl_table(model, tokenizer, data_loader, device=None):
    """
    Calculate the average perplexity for the chosen, rejected, and self-generated responses.
    Args:
        model: The model to use for scoring.
        tokenizer: The tokenizer to use for encoding the input.
        data_loader: DataLoader containing the dataset.
        device: Device to run the model on (default is model's device).
    Returns:
        dict: A dictionary with keys 'chosen', 'rejected', and 'self' corresponding to the average perplexity.
    """
    sum_chosen, sum_rejected, sum_self = 0.0, 0.0, 0.0
    count = 0
    eos = tokenizer.eos_token
    self_cache: Dict[str, float] = {}
    debug_printed = 0

    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating perplexity"):
            prompts = [tokenizer.decode(x, skip_special_tokens=False) for x in batch["prompt"]]
            chosens = [tokenizer.decode(x, skip_special_tokens=True) for x in batch["chosen"]]
            rejecteds = [tokenizer.decode(x, skip_special_tokens=True) for x in batch["rejected"]]
            
            # Debugging
            for i in range(len(prompts)):
                if debug_printed < 2:
                    print(f"\n=== DEBUG SAMPLE {i + 1} ===")
                    print(f"FULL PROMPT:\n{prompts[i]}\n")
                    print(f"CHOSEN RAW:\n{chosens[i]}\n\n")
                    print(f"REJECTED RAW:\n{rejecteds[i]}\n\n")
                else: break
                debug_printed += 1

            bsz = len(prompts)

            # # 1) Perform batch computation of perplexity for the chosen and rejected sets
            # texts = [f"{p}{c}{eos}" for p, c in zip(prompts, chosens)] \
            #     + [f"{p}{r}{eos}" for p, r in zip(prompts, rejecteds)]
            # ppls = calculate_perplexity(model, tokenizer, texts, batch_size=len(texts), device=device)
            # chosen_batch_ppls = ppls[:bsz]
            # rejected_batch_ppls = ppls[bsz:]

            chosen_texts = [f"{p}{c}{eos}" for p, c in zip(prompts, chosens)]
            rejected_texts = [f"{p}{r}{eos}" for p, r in zip(prompts, rejecteds)]
            chosen_ppls = calculate_perplexity(
                model, tokenizer, chosen_texts,
                max_length=config.allowed_max_length,
                device=device,
                batch_size=len(chosen_texts)
            )
            rejected_ppls = calculate_perplexity(
                model, tokenizer, rejected_texts,
                max_length=config.allowed_max_length,
                device=device,
                batch_size=len(rejected_texts)
            )

            # 2) batch self-generation + PPL
            input_ids = batch["prompt"].to(device)
            gen_outs = model.generate(
                input_ids,
                max_new_tokens=config.max_new_tokens,
                do_sample=False
            )
            resps = [postprocess_response(tokenizer.decode(o, skip_special_tokens=False)) for o in gen_outs]
            self_texts = [f"{p}{r}{eos}" for p, r in zip(prompts, resps)]
            self_ppls = calculate_perplexity(
                model, tokenizer, self_texts,
                max_length=config.allowed_max_length,
                device=device,
                batch_size=len(self_texts)
            )

            # accumulate with cache
            for prompt, chosen_batch_ppl, rejected_batch_ppl, self_ppl in zip(prompts, chosen_ppls, rejected_ppls, self_ppls):
                sum_chosen += chosen_batch_ppl
                sum_rejected += rejected_batch_ppl
                # cache self-PPL by prompt
                if prompt not in self_cache:
                    self_cache[prompt] = self_ppl
                sum_self += self_cache[prompt]
                count += 1

    if count == 0:
        raise ValueError("No samples processed")
    return {
        "chosen": sum_chosen / count,
        "rejected": sum_rejected / count,
        "self": sum_self / count,
    }


# ------------------------------- GPU Memory Management -------------------------------
def get_gpu_memory_usage():
    """
    Get the current GPU memory usage.

    Returns:
        tuple: (allocated_memory_GB, cached_memory_GB, total_memory_GB)
    """
    if not torch.cuda.is_available():
        return (0, 0, 0)

    device = torch.cuda.current_device()

    # Get memory statistics
    allocated_bytes = torch.cuda.memory_allocated(device)
    cached_bytes = torch.cuda.memory_reserved(device)
    total_bytes = torch.cuda.get_device_properties(device).total_memory

    # Convert to GB
    allocated_gb = allocated_bytes / (1024 ** 3)
    cached_gb = cached_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)

    return (allocated_gb, cached_gb, total_gb)


# ------------------------------- Print GPU Memory Usage -------------------------------
def print_gpu_memory_usage(prefix=""):
    """
    Print the current GPU memory usage.

    Args:
        prefix (str): Optional prefix to add before the memory usage output
    """
    if not torch.cuda.is_available():
        print(f"{prefix}GPU not available")
        return

    allocated_gb, cached_gb, total_gb = get_gpu_memory_usage()

    print(f"{prefix}GPU Memory: {allocated_gb:.2f}GB allocated, "
          f"{cached_gb:.2f}GB cached, "
          f"{total_gb:.2f}GB total, "
          f"{(allocated_gb/total_gb)*100:.1f}% used")


# ------------------------------- Log Memory Snapshot -------------------------------
def log_memory_snapshot(step_name=""):
    """
    Log a memory snapshot with a descriptive step name.

    Args:
        step_name (str): Name of the current step or operation
    """
    if not torch.cuda.is_available():
        return

    print(f"[MEMORY] {step_name} - ", end="")
    print_gpu_memory_usage()

    # Optional: force garbage collection
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()
