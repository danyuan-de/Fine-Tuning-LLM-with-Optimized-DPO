# model_name = "meta-llama/Llama-3.2-1B"
model_name = "meta-llama/Llama-3.1-8B"
fine_tuned_model_path = f"/workspace/{model_name.split('/')[-1]}_fine-tuned"
cache_dir = "/workspace/models" # for cloud storage

allowed_max_length = 512 # maximum number of tokens in a sequence for training input data
max_new_tokens = 128  # maximum number of tokens to generate
batch_size = 4
num_epochs = 1
beta = 0.1
learning_rate = 1e-5 # try 5e-5, 5e-6 or 1e-6, etc.
temperature = 0.7 # between 0.7 and 1.0, lower values generate more deterministic text
top_p = 0.8 # between 0.7 and 0.95, higher values generate more diverse text

file_content = "physics_qa_content.json"
file_structure = "physics_qa_structure.json"
file_preference = "instruction-data-with-preference.json"
