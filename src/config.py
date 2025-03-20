import os

# ---------------------------------- Directory paths ----------------------------------
# model_name = "meta-llama/Llama-3.2-1B"
model_name = "meta-llama/Llama-3.1-8B"
model_workspace_dir = os.path.join(os.path.dirname(__file__), "..", "workspace")
fine_tuned_model_path = os.path.join(model_workspace_dir, f"{model_name.split('/')[-1]}_fine-tuned")
cache_dir = os.path.join(model_workspace_dir, "models")

eot_token = "<|eot_id|>"  # End of text token for generation (Llama model)

# ---------------------------------- DPO loss parameters ----------------------------------
beta = 0.1 # Temperature parameter controlling the sharpness of the decision boundary in DPO loss
lambda_kl = 0.1 # Weight of the KL divergence penalty to prevent model drift in DPO loss

# ---------------------------------- Training parameters ----------------------------------
allowed_max_length = 1024 # maximum number of tokens in a sequence for training input data
max_new_tokens = 256  # maximum number of tokens to generate

batch_size = 8 # Process the number of items at once
num_epochs = 1 # Number of times to go through the dataset

learning_rate = 5e-6 # try 5e-5 or 5e-6, etc.
weight_decay = 0.001 # Weight decay for regularization, try 0.01, 0.001, etc. (0.01 may be too high)

temperature = 0.7 # between 0.7 and 1.0, lower values generate more deterministic text
top_p = 0.8 # between 0.7 and 0.95, higher values generate more diverse text

# ---------------------------------- Evaluation parameters ----------------------------------
early_stopping_patience = 3 # Stop training if validation loss doesn't improve for this many evaluations
max_reward_margin = 5.0 # Maximum allowed reward margin before stopping training

# ---------------------------------- Data paths ----------------------------------
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
file_preference = os.path.join(data_dir, "instruction-data-with-preference.json")
file_content = os.path.join(data_dir, "physics_qa_content.json")
file_structure = os.path.join(data_dir, "physics_qa_structure.json")
file_mixed = os.path.join(data_dir, "physics_qa_mixed.json")

# ------------------------ Results directory ------------------------
result_dir = os.path.join(os.path.dirname(__file__), "..", "results")

# ------------------------ Distributed training settings ------------------------
# Default NCCL backend for distributed training
# This is the most efficient for GPU-based distributed training
distributed_backend = "nccl"

# Port used for distributed training communication
# Can be overridden via environment variable MASTER_PORT
distributed_port = "12355"

# Whether to use mixed precision (fp16) training
# Can speed up training and reduce memory usage
use_mixed_precision = True  

# Gradient accumulation steps
# Increases effective batch size without increasing memory usage
gradient_accumulation_steps = 1  

# Whether to use ZeRO optimizer for memory efficiency
# Options: 0 (disabled), 1, 2, or 3 (most memory efficient but slower)
zero_optimization_level = 0

# Whether to apply gradient checkpointing
# Saves memory at the cost of additional computation
use_gradient_checkpointing = False
