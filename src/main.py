# Description: This script trains a model using DPO on the instruction data with preferences.
# Execute: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python -m src.main.py for MPS (MacOS)
# Execute: python -m src.main for CUDA (Linux)
# Update pytorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/mps
from src.argsParse import parse_args, update_config_from_args, print_configuration
from src.utils import load_split_data


def main():
    # ----------------------------------- Argument Parsing -----------------------------------
    args = parse_args()  # Parse command-line arguments
    update_config_from_args(args)  # Update config with parsed arguments
    print_configuration()  # Print the configuration

    train_data, val_data, test_data, datalength = load_split_data()

    if args.sft:
        from src.sft_trainer import run_sft
        print("Running supervised fine-tuning (SFT)...")
        run_sft(train_data)
        print("SFT completed.")
    if args.train:
        from src.trainer import run_training
        print("Running training...")
        run_training(train_data, val_data, test_data, datalength)
    if args.benchmark:
        from src.benchmark_test import run_benchmark
        print("Running benchmark test...")
        run_benchmark()
        print("Benchmark completed.")
    else:
        print("No action specified. Use --train to run training or --benchmark to run the benchmark test.")


if __name__ == "__main__":
    main()
