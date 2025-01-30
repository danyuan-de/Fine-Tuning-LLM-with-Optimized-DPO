import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch

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
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        ) # for the instruction-data-with-preference.json
    elif "question" in entry:
        instruction_text = (
            f"Answer the following question using step-by-step reasoning and the appropriate equations."
            f"\n\n### Instruction:\n{entry['question']}"
        )
    # input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else "" # for the instruction-data-with-preference.json
    return instruction_text #+ input_text

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
    plt.savefig(f"{label}-plot.pdf")
    # plt.show()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text) # , allowed_special={"<|endoftext|>"}) --> it's OpenAI's tiktoken
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size=512, temperature=0.0, top_k=None, eos_id=None):
    """
    Generate text using the model.

    Args:
        model: The language model.
        idx: Input token IDs (batch_size, seq_len).
        max_new_tokens: Maximum number of tokens to generate.
        context_size: Context size for the model.
        temperature: Temperature for sampling.
        top_k: Top-k sampling parameter.
        eos_id: End-of-sequence token ID.

    Returns:
        idx: Generated token IDs (batch_size, seq_len + max_new_tokens).
    """
    for _ in range(max_new_tokens):
        # Truncate the input to the context size
        idx_cond = idx[:, -context_size:]

        # Get the model's output
        with torch.no_grad():
            output = model(idx_cond)

        # Extract the logits from the output
        if hasattr(output, "logits"):
            logits = output.logits  # Extract logits from CausalLMOutputWithPast
        elif isinstance(output, tuple):
            logits = output[0]  # Extract logits from a tuple
        else:
            logits = output  # Assume output is already logits

        # Focus on the last time step
        logits = logits[:, -1, :]

        # Apply top-k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Greedy decoding (no sampling)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Stop generating if the end-of-sequence token is encountered
        if idx_next == eos_id:
            break

        # Append the generated token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens + 1)

    return idx

