# model_name = "meta-llama/Llama-3.2-1B"
model_name = "meta-llama/Llama-3.1-8B"
fine_tuned_model_path = f"/workspace/{model_name.split('/')[-1]}_fine-tuned"
cache_dir = "/workspace/models" # for cloud storage

batch_size = 4

file_content = "physics_qa_content.json"
file_structure = "physics_qa_structure.json"