import os


# -------------------------- Run benchmark --------------------------
benchmark = False  # Set to True to run the benchmark
train = False  # Set to True to run the training

benchmark_datasets = {
    1: "Eureka-Lab/PHYBench",
    2: "TIGER-Lab/MMLU-Pro"
}

benchmark_dataset = benchmark_datasets[1]  # default dataset for benchmark
num_benchmark_samples = 100  # Number of samples to benchmark, set to 0 for all samples
MMLU_PRO_category_isPhysics = False  # Set to True to run the benchmark on the physics dataset

# ---------------------------------- Random seed ----------------------------------
random_seed = 42  # Seed for reproducibility

# ---------------------------------- Model parameters ----------------------------------
models = {
    "8B": "meta-llama/Llama-3.1-8B",
    "8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "8B-SFT": "prithivMLmods/Llama-3.1-8B-Open-SFT",
    "PhyMaster": "gallen881/Llama-3-8B-Physics_Master-GGUF"
}

model_name = models["8B-SFT"]  # default model

# ---------------------------------- Directory paths ----------------------------------
model_workspace_dir = os.path.join(os.path.dirname(__file__), "..", "workspace")
fine_tuned_model_path = os.path.join(model_workspace_dir, f"{model_name.split('/')[-1]}_fine-tuned")
cache_dir = os.path.join(model_workspace_dir, "models")

# ---------------------------------- Method ----------------------------------
methods = {
    "DPO": "dpo",
    "DPOP": "dpop",
    "sDPO": "dposhift",
    "sDPOP": "dpopshift"
}

method_name = methods["DPO"]  # default method

# ---------------------------------- DPO parameters ----------------------------------
beta = 0.3  # Temperature parameter controlling the sharpness of the decision boundary in DPO loss
lambda_dpop = 50.0  # Weight for DPOP term to prevent reduction of preferred completion likelihood
lambda_shift = 0.9  # Weight for the shift term in DPO loss

# ---------------------------------- Model parameters ----------------------------------
allowed_max_length = 4096  # maximum number of tokens in a sequence for training input data
max_new_tokens = 512  # maximum number of tokens to generate

# --------------------------------- Training parameters ---------------------------------
batch_size = 2  # Process the number of items at once
gradient_accumulation_steps = 8  # Number of steps to accumulate gradients before stepping
num_epochs = 1  # Number of times to go through the dataset

# Some studies suggest using a learning rate between 5e-7 and 1e-7 for DPO but too low for DPOP
learning_rate = 5e-7

weight_decay = 0.01  # Original: 0.001 - Higher regularization to prevent overfitting

EVAL_USE_SAMPLING = False  # Use sampling for evaluation
temperature = 0.7  # between 0.7 and 1.0, lower values generate more deterministic text
top_p = 0.9  # between 0.7 and 0.95, higher values generate more diverse text

# ---------------------------------- Evaluation parameters ----------------------------------
eval_freq = 5  # Frequency of evaluation during training

# ---------------------------------- Data paths ----------------------------------
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
training_data_files = {
    'content': os.path.join(data_dir, "physics_qa_content.json"),
    'structure': os.path.join(data_dir, "physics_qa_structure.json"),
    'html': os.path.join(data_dir, "physics_qa_html.json"),
    'mixed': os.path.join(data_dir, "physics_qa_mixed.json"),
    'preference': os.path.join(data_dir, "instruction-data-with-preference.json")
}

training_data_filename = training_data_files['html']  # default training data

# ------------------------ Results directory ------------------------
result_dir = os.path.join(os.path.dirname(__file__), "..", "results")

# ------------------------ Prompt template for training and PHYBench ------------------------
system_prompt_physics = (
    "You are a physics expert assistant. "
    "Provide a concise reasoning process followed by a clear final answer for the following question."
)

# -------------------------- Prompt template for MMLU-Pro Benchmark --------------------------
system_prompt_mc_general = (
    "You are a helpful, knowledgeable assistant. "
    "Carefully analyze the following multiple-choice question and options. "
    "Provide a reasoning process to determine the correct answer. "
    "End your response in the format 'Final Answer: $\\boxed{X}$', where X is the letter (A, B, C, D, ...) of the correct option."
)

system_prompt_mc_physics = (
    "You are a physics expert assistant. "
    "Carefully analyze the following multiple-choice question and options. "
    "Provide a reasoning process to determine the correct answer. "
    "End your response in the format 'Final Answer: $\\boxed{X}$', where X is the letter (A, B, C, D, ...) of the correct option."
)
