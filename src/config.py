import os

# ---------------------------------- Model parameters ----------------------------------
models = {
    "1B": "meta-llama/Llama-3.2-1B",
    "1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "8B": "meta-llama/Llama-3.1-8B",
    "8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct"
}

model_name = models["8B-Instruct"] # default model

# ---------------------------------- Directory paths ----------------------------------
model_workspace_dir = os.path.join(os.path.dirname(__file__), "..", "workspace")
fine_tuned_model_path = os.path.join(model_workspace_dir, f"{model_name.split('/')[-1]}_fine-tuned")
cache_dir = os.path.join(model_workspace_dir, "models")

# ---------------------------------- Method ----------------------------------
methods = {
    "DPO": "dpo",
    "DPOP": "dpop",
    "DPOKL": "dpokl",
    "DPOPKL": "dpopkl",
    "DPOC": "dpocontrast"
}

method_name = methods["DPO"] # default method

# ---------------------------------- DPO parameters ----------------------------------
beta = 0.3  # Temperature parameter controlling the sharpness of the decision boundary in DPO loss
lambda_kl = 0.001  # Weight of the KL divergence penalty to prevent model drift in DPO loss
lambda_dpop = 50.0  # Weight for DPOP term to prevent reduction of preferred completion likelihood
lambda_contrast = 0.01  # Weight for contrastive loss term to encourage diversity in generated samples

# ---------------------------------- Model parameters ----------------------------------
allowed_max_length = 4096 # maximum number of tokens in a sequence for training input data
max_new_tokens = 256  # maximum number of tokens to generate

# --------------------------------- Training parameters ---------------------------------
batch_size = 2  # Process the number of items at once
gradient_accumulation_steps = 8  # Number of steps to accumulate gradients before stepping
num_epochs = 3  # Number of times to go through the dataset

# Some studies suggest using a learning rate between 5e-7 and 1e-7 for DPO but too low for DPOP
learning_rate = 1e-6
# Tried 1e-6 and 3e-6 for DPOP, still seems low to learn the policy model output are almost the same as the reference model

weight_decay = 0.01  # Original: 0.001 - Higher regularization to prevent overfitting

temperature = 0.7 # between 0.7 and 1.0, lower values generate more deterministic text
top_p = 0.8 # between 0.7 and 0.95, higher values generate more diverse text

# ---------------------------------- Evaluation parameters ----------------------------------
early_stopping_patience = 3 # Stop training if validation loss doesn't improve for this many evaluations
eval_freq = 2  # Original: 5 - Evaluate more frequently to catch divergence earlier
max_reward_margin = 5.0 # Maximum allowed reward margin before stopping training

# ---------------------------------- Data paths ----------------------------------
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
training_data_files = {
    'content': os.path.join(data_dir, "physics_qa_content.json"),
    'structure': os.path.join(data_dir, "physics_qa_structure.json"),
    'html': os.path.join(data_dir, "physics_qa_html.json"),
    'mixed': os.path.join(data_dir, "physics_qa_mixed.json"),
    'preference': os.path.join(data_dir, "instruction-data-with-preference.json")
}

training_data_filename = training_data_files['html'] # default training data

# ------------------------ Results directory ------------------------
result_dir = os.path.join(os.path.dirname(__file__), "..", "results")