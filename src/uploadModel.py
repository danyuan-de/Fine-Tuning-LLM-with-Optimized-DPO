from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import src.config as config

repo_id = "DanYuDE/llama-3.1-8B-dpo"  # Hugging Face Repository ID
save_directory = config.fine_tuned_model_path  # model path

# load fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)

# push model and tokenizer to Hugging Face Hub
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
