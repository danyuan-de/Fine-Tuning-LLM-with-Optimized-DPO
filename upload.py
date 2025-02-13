from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import config

repo_id = "DanYuDE/llama-3.1-8B-dpo"  # Hugging Face Repository ID
save_directory = config.fine_tuned_model_path  # model path

# 加載 Fine-Tuned Model 和 Tokenizer
model = AutoModelForCausalLM.from_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)

# 上傳模型
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
