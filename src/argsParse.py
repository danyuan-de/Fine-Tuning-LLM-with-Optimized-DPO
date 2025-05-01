import argparse
import src.config as config


# This script is used to parse command line arguments for training a model.
def parse_args():
    """
    Parse command line arguments for the DPO training script.
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Train a model using DPO with custom hyperparameters')

    # ---------------------- Select a flag to run training or benchmark test ------------------------
    parser.add_argument('--train', action='store_true', default=False,
                        help='Run training on the dataset')
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='Run benchmark on the dataset')

    # -------------------------------- Select benchmark test dataset --------------------------------
    benchmark_choices = list(config.benchmark_datasets.keys())
    benchmark_help = "Benchmark dataset choice: " + ", ".join([f"{key}: {value}" for key, value in config.benchmark_datasets.items()])
    parser.add_argument('--benchmark_dataset', type=int, choices=benchmark_choices, default=1,
                        help=benchmark_help)
    parser.add_argument('--num_benchmark_samples', type=int, default=config.num_benchmark_samples,
                        help='Number of samples to benchmark')
    parser.add_argument('--category_isPhysics', action='store_true', default=False,
                        help='Run MMLU-Pro benchmark on the physics category')

    # ---------------------------------- Random seed for reproducibility ---------------------------
    parser.add_argument('--seed', type=int, default=config.random_seed,
                        help='Random seed for reproducibility')

    # --------------------- Model selection - directly select the model name ---------------------
    model_choices = list(config.models.keys())
    model_help = "Model choice: " + ", ".join(model_choices)
    parser.add_argument('--model', type=str, choices=model_choices, default="8B-SFT",
                        help=model_help)

    # -------------------- Method selection - directly select the method name --------------------
    method_choices = list(config.methods.keys())
    method_help = "Method choice: " + ", ".join(method_choices)
    parser.add_argument('--method', type=str, choices=method_choices, default="sDPO",
                        help=method_help)

    # ------------------------------------ DPO loss parameters ------------------------------------
    parser.add_argument('--beta', type=float, default=config.beta,
                        help='Beta value for DPO loss')
    parser.add_argument('--lambda_dpop', type=float, default=config.lambda_dpop,
                        help='Lambda DPOP value')
    parser.add_argument('--lambda_shift', type=float, default=config.lambda_shift,
                        help='Lambda shift value')

    # ------------------------------------ Training parameters ------------------------------------
    parser.add_argument('--lr', type=float, default=config.learning_rate,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=config.batch_size,
                        help='Batch size')
    parser.add_argument('--grad_accum', type=int, default=config.gradient_accumulation_steps,
                        help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=config.num_epochs,
                        help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=config.weight_decay,
                        help='Weight decay')
    parser.add_argument('--max_length', type=int, default=config.allowed_max_length,
                        help='Maximum input length')
    parser.add_argument('--max_new_tokens', type=int, default=config.max_new_tokens,
                        help='Maximum tokens to generate')

    # ------------------------------------ Generation parameters ------------------------------------
    parser.add_argument('--sampling', action='store_true', default=config.EVAL_USE_SAMPLING,
                        help='Use sampling for evaluation')
    parser.add_argument('--temp', type=float, default=config.temperature,
                        help='Temperature for generation')
    parser.add_argument('--top_p', type=float, default=config.top_p,
                        help='Top-p sampling parameter')

    # -------------------- Training data parameters - use the mapping from config --------------------
    parser.add_argument('--data', type=str, choices=list(config.training_data_files.keys()),
                        default='html', help='Data choice (content, structure, html, mixed, preference)')

    # Optional: Add a direct file path option for more flexibility
    parser.add_argument('--data_file', type=str, default=None,
                        help='Direct path to data file (overrides --data if specified)')

    # ------------------------------------ Evaluation parameters ------------------------------------
    parser.add_argument('--eval_freq', type=int, default=config.eval_freq,
                        help='Evaluation frequency')

    # Parse the arguments
    args = parser.parse_args()

    return args


def update_config_from_args(args):
    """
    Update configuration values based on parsed command-line arguments.

    Args:
        args (argparse.Namespace): The parsed command-line arguments
    """
    # Update config values with command-line arguments
    config.train = args.train
    config.benchmark = args.benchmark
    config.benchmark_dataset = args.benchmark_dataset
    config.num_benchmark_samples = args.num_benchmark_samples
    config.MMLU_PRO_category_isPhysics = args.category_isPhysics
    config.random_seed = args.seed
    config.beta = args.beta
    config.lambda_dpop = args.lambda_dpop
    config.lambda_shift = args.lambda_shift
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.gradient_accumulation_steps = args.grad_accum
    config.num_epochs = args.epochs
    config.weight_decay = args.weight_decay
    config.allowed_max_length = args.max_length
    config.max_new_tokens = args.max_new_tokens
    config.EVAL_USE_SAMPLING = args.sampling
    config.temperature = args.temp
    config.top_p = args.top_p
    config.eval_freq = args.eval_freq
    config.model_name = config.models[args.model]
    config.method_name = config.methods[args.method]

    # Be able to specify a direct file path or use the mapping
    config.training_data_filename = args.data_file if args.data_file else config.training_data_files[args.data]


def print_configuration():
    """
    Print the current training configuration.

    Args:
        args: The parsed arguments
    """
    print(f"\n{'='*50}")
    print("TRAINING CONFIGURATION:")
    print(f"{'='*50}")
    print(f"Random Seed: {config.random_seed}")
    if config.train:
        print(f"Run Training: {config.train}")
        print(f"Model: {config.model_name}")
        print(f"Method: {config.method_name.upper()}")
        print(f"Training Data: {config.training_data_filename}")
        print("\nDPO Parameters:")
        print(f"  Beta: {config.beta}")
        if config.method_name in ['dpop', 'dpopshift']:
            print(f"  Lambda DPOP: {config.lambda_dpop}")
        if config.method_name in ['dposhift', 'dpopshift']:
            print(f"  Lambda Shift: {config.lambda_shift}")
        print("\nTraining Parameters:")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Gradient Accumulation Steps: {config.gradient_accumulation_steps}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Weight Decay: {config.weight_decay}")
        print(f"  Evaluation Frequency: {config.eval_freq}")
        print("\nModel Parameters:")
        print(f"  Max Input Length: {config.allowed_max_length}")
        print(f"  Max New Tokens: {config.max_new_tokens}")
        print("\nGeneration Parameters:")
        print(f"  Use Sampling: {config.EVAL_USE_SAMPLING}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Top-p: {config.top_p}")

    elif config.benchmark:
        print(f"Run Benchmark: {config.benchmark}")
        print(f"Benchmark Dataset: {config.benchmark_dataset}")
        print(f"Number of Benchmark Samples: {config.num_benchmark_samples}")
        print(f"Category is Physics: {config.MMLU_PRO_category_isPhysics}")
        print(f"use Sampling: {config.EVAL_USE_SAMPLING}")
    print(f"{'='*50}\n")
