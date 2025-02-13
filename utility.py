import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import re
from transformers import StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM, AutoTokenizer
import math

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
            "Deliver accurate and concise thinking processes and answers based on the following question."
        )
        return (
            "<|begin_of_text|>\n"
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"Question: {entry['question']}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )

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

def generate(
    model, 
    idx, 
    stopping_criteria=None,
    max_new_tokens=100,
    context_size=512,
    temperature=0.3,
    top_k=None,
    top_p=None,  
    eot_token_id=None
):
    """
    enhanced generation function that supports both top-k and top-p
    priority: top_p > top_k (when both are set)
    """
    for _ in range(max_new_tokens):
        # truncate context
        idx_cond = idx[:, -context_size:].to(model.device) 
        
        # get logits
        with torch.no_grad():
            output = model(idx_cond)
            logits = output.logits if hasattr(output, "logits") else output[0]
        
        # select the last token's logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        
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
        idx_next = torch.multinomial(probs, num_samples=1).to(model.device)

        # EOT check
        if idx_next == eot_token_id:
            break

        idx = torch.cat((idx, idx_next), dim=1).to(model.device)

    return idx

class EOTStoppingCriteria(StoppingCriteria):
    def __init__(self, eot_token_id):
        self.eot_token_id = eot_token_id
        
    def __call__(self, input_ids, scores, **kwargs):
        # check if the last token is the EOT token
        return input_ids[0][-1] == self.eot_token_id
    

# postprocess response to remove unwanted tokens
def postprocess_response(full_text: str) -> str:
    # make sure only the first assistant part is kept
    if "<|start_header_id|>assistant<|end_header_id|>" in full_text:
        response = full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        # remove all subsequent HTML tags
        response = re.sub(r"<[^>]+>", "", response)
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
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids#.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    perplexity = math.exp(loss.item())
    return perplexity