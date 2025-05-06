# Setup script for root privilege

#!/bin/bash

# Make the script executable with the following command:
# chmod +x setup.sh
# Run the script with arguments, example:
# ./setup.sh --train --beta 0.1 --lambda_dpop 50.0 --lr 5e-6
# ./setup.sh --benchmark --model 8B-Instruct --temp 0.7

# Display help information if requested
if [[ "$1" == "--help" ]]; then
  echo "Usage: ./setup.sh [options]"
  echo ""
  echo "This script sets up the environment and runs the DPO training or benchmark with specified parameters."
  echo ""
  echo "Options:"
  echo "  Mode Selection:"
  echo "    --train               Run training on the dataset (default: False)"
  echo "    --benchmark           Run benchmark test (default: False)"
  echo ""
  echo "  Benchmark dataset selection:"
  echo "    --benchmark_dataset N     Benchmark dataset choice (1: cais/mmlu, 2: TIGER-Lab/MMLU-Pro, 3: Eureka-Lab/PHYBench) (default: 1)"
  echo "    --category_isPhysics      Run MMLU-Pro benchmark on the physics category (default: False)"
  echo "    --num_benchmark_samples N Number of samples to benchmark (default: 100)"
  echo ""
  echo "  Random Seed:"
  echo "    --seed N              Random seed for reproducibility (default: 42)"
  echo ""
  echo "  Model Selection:"
  echo "    --model MODEL         Model choice (1B, 1B-Instruct, 8B, 8B-Instruct, 8B-SFT, PhyMaster) (default: 1B-Instruct)"
  echo ""
  echo "  Method Selection:"
  echo "    --method METHOD       Method choice (DPO, DPOP, sDPO, sDPOP) (default: sDPO)"
  echo ""
  echo "  Log-probs Option:"
  echo "    --avglps              Use average log probabilities for DPO loss (default: False)"
  echo ""
  echo "  DPO Loss Parameters:"
  echo "    --beta VALUE          Beta value for DPO loss (default: 0.3)"
  echo "    --lambda_dpop VALUE   Lambda DPOP value (default: 50.0)"
  echo "    --lambda_shift VALUE  Lambda shift value (default: 0.75)"
  echo ""
  echo "  Training Parameters:"
  echo "    --lr VALUE            Learning rate (default: 5e-7)"
  echo "    --batch_size N        Batch size (default: 2)"
  echo "    --grad_accum N        Gradient accumulation steps (default: 8)"
  echo "    --epochs N            Number of epochs (default: 1)"
  echo "    --weight_decay VALUE  Weight decay (default: 0.01)"
  echo "    --max_length N        Maximum input length (default: 4096)"
  echo "    --max_new_tokens N    Maximum tokens to generate (default: 512)"
  echo ""
  echo "  Generation Parameters:"
  echo "    --sampling            Use sampling for evaluation (default: False)"
  echo "    --temp VALUE          Temperature for generation (default: 0.7)"
  echo "    --top_p VALUE         Top-p sampling parameter (default: 0.9)"
  echo ""
  echo "  Data Parameters:"
  echo "    --data TYPE           Data choice (content, structure, html, mixed, chat, preference, data) (default: data)"
  echo "    --data_file PATH      Direct path to data file (overrides --data if specified)"
  echo ""
  echo "  Evaluation Parameters:"
  echo "    --eval_freq N         Evaluation frequency (default: 5)"
  echo ""
  echo "Example:"
  echo "  ./setup.sh --train --model 8B-SFT --method DPOP \\"
  echo "               --beta 0.2 --lambda_dpop 30.0 --data mixed"
  exit 0
fi


# Stop execution on any error
set -e

# Update package lists
echo "Updating package lists..."
apt-get update

# Install Python virtual environment package and nano
echo "Installing python3.10-venv and nano..."
apt-get install -y python3.10-venv nano

# Create a virtual environment if it doesn't exist
VENV_DIR="env"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip and install Hugging Face CLI
echo "Installing Hugging Face CLI..."
pip install -U "huggingface_hub[cli]"

# Log into Hugging Face CLI
echo "Please log into Hugging Face CLI (Press Enter when ready)"
huggingface-cli login

# Check if `requirements.txt` exists and install dependencies
REQUIREMENTS_FILE="requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "No requirements.txt found. Skipping dependency installation."
fi

# Check for CUDA availability and set appropriate environment variables
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "CUDA detected, enabling GPU acceleration..."
    export CUDA_VISIBLE_DEVICES=0
else
    # For MacOS with Apple Silicon
    if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
        echo "Apple Silicon detected, enabling MPS acceleration..."
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    else
        echo "No GPU detected, using CPU only."
    fi
fi

# Detect mode flags
TRAIN_MODE=false
BENCHMARK_MODE=false
for arg in "$@"; do
  case "$arg" in
    --train) TRAIN_MODE=true ;;
    --benchmark) BENCHMARK_MODE=true ;;
  esac
done

# Run the main logic
PYTHONPATH=$(pwd) python -m src.main "$@"

# Helper to print the Hugging Face hint
print_upload_hint() {
  echo "Model saved to the workspace. You can upload it to the Hugging Face Hub with:"
  echo "  python -m src.uploadModel"
}

# Print a message based on the mode
if $TRAIN_MODE && ! $BENCHMARK_MODE; then
  echo ">>> Completed in TRAIN mode"
  print_upload_hint
elif $BENCHMARK_MODE && ! $TRAIN_MODE; then
  echo ">>> Completed in BENCHMARK mode"
  print_upload_hint
elif $TRAIN_MODE && $BENCHMARK_MODE; then
  echo ">>> Completed both TRAIN and BENCHMARK"
  print_upload_hint
else
  echo ">>> No mode specified; script finished"
fi