import argparse
import src.config as config

# This script is used to parse command line arguments for training a model using DPO (Direct Performance Optimization).

def parse_args():
    """
    Parse command line arguments for the DPO training script.
    
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train a model using DPO with custom hyperparameters')

    # ------------------------------------ DPO loss parameters ------------------------------------
    parser.add_argument('--beta', type=float, default=config.beta, 
                        help='Beta value for DPO loss')
    parser.add_argument('--lambda_dpop', type=float, default=config.lambda_dpop, 
                        help='Lambda DPOP value')
    parser.add_argument('--lambda_kl', type=float, default=config.lambda_kl, 
                        help='Lambda KL value')
    parser.add_argument('--lambda_contrast', type=float, default=config.lambda_contrast, 
                        help='Lambda contrast value')
    
    # --------------------- Model selection - directly select the model name ---------------------
    model_choices = list(config.models.keys())
    model_help = "Model choice: " + ", ".join(model_choices)
    parser.add_argument('--model', type=str, choices=model_choices, default="8B-Instruct",
                        help=model_help)

    # -------------------- Method selection - directly select the method name --------------------
    method_choices = list(config.methods.keys())
    method_help = "Method choice: " + ", ".join(method_choices)
    parser.add_argument('--method', type=str, choices=method_choices, default="DPO", 
                        help=method_help)

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
    parser.add_argument('--temp', type=float, default=config.temperature, 
                        help='Temperature for generation')
    parser.add_argument('--top_p', type=float, default=config.top_p, 
                        help='Top-p sampling parameter')

    # -------------------- Training data parameters - use the mapping from config --------------------
    parser.add_argument('--data', type=str, choices=list(config.data_files.keys()), 
                        default='content', help='Data choice (content, structure, mixed, preference)')
                        
    # Optional: Add a direct file path option for more flexibility
    parser.add_argument('--data_file', type=str, default=None,
                        help='Direct path to data file (overrides --data if specified)')

    # ------------------------------------ Evaluation parameters ------------------------------------
    parser.add_argument('--eval_freq', type=int, default=config.eval_freq, 
                        help='Evaluation frequency')
    parser.add_argument('--eval_patience', type=int, 
                        default=config.early_stopping_patience if hasattr(config, 'early_stopping_patience') else 3, 
                        help='Early stopping patience')

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
    config.beta = args.beta
    config.lambda_dpop = args.lambda_dpop
    config.lambda_kl = args.lambda_kl
    config.lambda_contrast = args.lambda_contrast
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.gradient_accumulation_steps = args.grad_accum
    config.num_epochs = args.epochs
    config.weight_decay = args.weight_decay
    config.allowed_max_length = args.max_length
    config.max_new_tokens = args.max_new_tokens
    config.temperature = args.temp
    config.top_p = args.top_p
    config.eval_freq = args.eval_freq

    config.model_name = config.models[args.model]
    config.method_name = config.methods[args.method]
    config.training_data_file = args.data_file if args.data_file else config.data_files[args.data] # You can specify a direct file path or use the mapping
    
    # Update early stopping patience if it exists in config
    if hasattr(config, 'early_stopping_patience'):
        config.early_stopping_patience = args.eval_patience

def print_configuration():
    """
    Print the current training configuration.
    
    Args:
        args: The parsed arguments
    """
    print(f"\n{'='*50}")
    print(f"TRAINING CONFIGURATION:")
    print(f"{'='*50}")
    print(f"Model: {config.model_name}")
    print(f"Method: {config.method_name.upper()}")
    print(f"Data: {config.training_data_file}")
    print(f"\nDPO Parameters:")
    print(f"  Beta: {config.beta}")
    if config.method_name in ['dpop', 'dpopkl']:
        print(f"  Lambda DPOP: {config.lambda_dpop}")
    if config.method_name in ['dpokl', 'dpopkl']:
        print(f"  Lambda KL: {config.lambda_kl}")
    if config.method_name == 'dpocontrast':
        print(f"  Lambda Contrast: {config.lambda_contrast}")
    print(f"\nTraining Parameters:")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Gradient Accumulation Steps: {config.gradient_accumulation_steps}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Weight Decay: {config.weight_decay}")
    print(f"  Evaluation Frequency: {config.eval_freq}")
    if hasattr(config, 'early_stopping_patience'):
        print(f"  Early Stopping Patience: {config.early_stopping_patience}")
    print(f"\nModel Parameters:")
    print(f"  Max Input Length: {config.allowed_max_length}")
    print(f"  Max New Tokens: {config.max_new_tokens}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Top-p: {config.top_p}")
    print(f"{'='*50}\n")