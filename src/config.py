import os

result_dir = os.path.join(os.path.dirname(__file__), "..", "results")
output_text = os.path.join(result_dir, "output_test.txt")

# model_name = "meta-llama/Llama-3.2-1B"
model_name = "meta-llama/Llama-3.1-8B"
fine_tuned_model_path = os.path.join(os.path.dirname(__file__), "..", "workspace", f"{model_name.split('/')[-1]}_fine-tuned")
cache_dir = os.path.join(os.path.dirname(__file__), "..", "workspace", "models")

beta = 0.1 # Temperature parameter controlling the sharpness of the decision boundary in DPO loss
lambda_kl = 0.1 # Weight of the KL divergence penalty to prevent model drift in DPO loss

allowed_max_length = 512 # maximum number of tokens in a sequence for training input data
max_new_tokens = 256  # maximum number of tokens to generate
batch_size = 4 # Process the number of items at once
num_epochs = 1 # Number of times to go through the dataset
learning_rate = 1e-5 # try 5e-5, 5e-6 or 1e-6, etc.
temperature = 0.7 # between 0.7 and 1.0, lower values generate more deterministic text
top_p = 0.8 # between 0.7 and 0.95, higher values generate more diverse text
early_stopping_patience = 3 # Stop training if validation loss doesn't improve for this many evaluations
max_reward_margin = 5.0 # Maximum allowed reward margin before stopping training

data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
file_preference = os.path.join(data_dir, "instruction-data-with-preference.json")
file_content = os.path.join(data_dir, "physics_qa_content.json")
file_structure = os.path.join(data_dir, "physics_qa_structure.json")
file_mixed = os.path.join(data_dir, "physics_qa_mixed.json")
