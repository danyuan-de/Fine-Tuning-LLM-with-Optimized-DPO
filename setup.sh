#!/bin/bash

# Make the script executable with the following command:
# chmod +x setup.sh
# Run the script with arguments, example:
# ./setup.sh --beta 0.1 --lambda_dpop 50.0 --lr 5e-6

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
  echo "    --lambda_shift VALUE  Lambda shift value (default: from config.py)"
  echo ""
  echo "  Method Selection:"
  echo "    --method METHOD       Method choice (DPO, DPOP, sDPO, sDPOP) (default: DPO)"
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
if command -v apt-get &>/dev/null; then
  echo "Updating package lists…"
  if apt-get update -qq; then
    echo "  → update succeeded without sudo"
  else
    echo "  → retrying with sudo"
    sudo apt-get update
  fi
fi

# Install Astral uv using official install script
if ! command -v uv >/dev/null; then
  echo "Installing Astral uv via official install script..."
  wget -qO- https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Create or reuse virtual environment (.venv)
VENV_DIR=".venv"
uv venv --directory "$VENV_DIR" --python python3.10

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Synchronize dependencies from requirements.txt
echo "Syncing dependencies from requirements.txt..."
uv pip sync requirements.txt

# Install or update Hugging Face CLI
echo "Installing Hugging Face CLI..."
uv pip install -U "huggingface_hub[cli]"

echo "Please log into Hugging Face CLI (press Enter when ready)"
uvx huggingface-cli login

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

# Run training with provided parameters
echo "Running training with provided parameters..."
uv run python -m src.main "$@"

# Completion message
echo "Training complete."
echo "Model saved to workspace directory. You can upload it to Hugging Face Hub using:"
echo "  python -m src.uploadModel"