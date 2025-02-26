# model_name = "meta-llama/Llama-3.2-1B"
model_name = "meta-llama/Llama-3.1-8B"
fine_tuned_model_path = f"/workspace/{model_name.split('/')[-1]}_fine-tuned"
cache_dir = "/workspace/models" # for cloud storage

allowed_max_length = 512 # maximum number of tokens in a sequence for training input data
max_new_tokens = 100 # maximum number of tokens to generate
batch_size = 4
num_epochs = 1
beta = 0.1
learning_rate = 5e-6

file_content = "physics_qa_content.json"
file_structure = "physics_qa_structure.json"