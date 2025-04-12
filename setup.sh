#!/bin/bash

# Make the script executable with the following command:
# chmod +x setup.sh
# Run the script with arguments, example:
# ./setup.sh --beta 0.1 --lambda_dpop 40.0 --lambda_kl 0.2 --lr 5e-6

# Display help information if requested
if [[ "$1" == "--help" ]]; then
  echo "Usage: ./setup.sh [options]"
  echo ""
  echo "This script sets up the environment and runs the DPO training with specified parameters."
  echo ""
  echo "Options:"
  echo "  Model Selection:"
  echo "    --model MODEL         Model choice (1B, 1B-Instruct, 8B, 8B-Instruct) (default: 8B-Instruct)"
  echo ""
  echo "  DPO Loss Parameters:"
  echo "    --beta VALUE          Beta value for DPO loss (default: from config.py)"
  echo "    --lambda_dpop VALUE   Lambda DPOP value (default: from config.py)"
  echo "    --lambda_kl VALUE     Lambda KL value (default: from config.py)"
  echo "    --lambda_contrast VAL Lambda contrast value (default: from config.py)"
  echo ""
  echo "  Method Selection:"
  echo "    --method METHOD       Method choice (DPO, DPOP, DPOKL, DPOPKL, DPOC) (default: DPO)"
  echo ""
  echo "  Training Parameters:"
  echo "    --lr VALUE            Learning rate (default: from config.py)"
  echo "    --batch_size VALUE    Batch size (default: from config.py)"
  echo "    --grad_accum VALUE    Gradient accumulation steps (default: from config.py)"
  echo "    --epochs VALUE        Number of epochs (default: from config.py)"
  echo "    --weight_decay VALUE  Weight decay (default: from config.py)"
  echo "    --warmup_steps VALUE  Warmup steps (default: from config.py)"
  echo "    --max_length VALUE    Maximum input length (default: from config.py)"
  echo "    --max_new_tokens VAL  Maximum tokens to generate (default: from config.py)"
  echo ""
  echo "  Generation Parameters:"
  echo "    --temp VALUE          Temperature for generation (default: from config.py)"
  echo "    --top_p VALUE         Top-p sampling parameter (default: from config.py)"
  echo ""
  echo "  Data Parameters:"
  echo "    --data TYPE           Data type (content, structure, html, mixed, preference) (default: html)"
  echo "    --data_file PATH      Direct path to data file (overrides --data if specified)"
  echo ""
  echo "  Evaluation Parameters:"
  echo "    --eval_freq VALUE     Evaluation frequency (default: from config.py)"
  echo ""
  echo "Example:"
  echo "  ./setup.sh --beta 0.2 --lambda_dpop 30.0 --method DPOP --data mixed"
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

# Pass all arguments directly to the Python script
echo "Running training with provided parameters..."
PYTHONPATH=$(pwd) python -m src.main "$@"

echo "Training complete."
echo "Model saved to workspace directory. You can upload it to Hugging Face Hub using:"
echo "python -m src.uploadModel"