#!/bin/bash

# Make the script executable with the following command:
# chmod +x run_distributed.sh
# Run the script with the following command:
# ./run_distributed.sh [NUM_GPUS]

# Stop execution on any error
set -e

# Default number of GPUs to use (can be overridden by argument)
NUM_GPUS=${1:-2}  # Default to 2 GPUs if not specified
# Use a random port to avoid conflicts (between 30000 and 40000)
MASTER_PORT=$((30000 + RANDOM % 10000))

echo "Starting distributed training with $NUM_GPUS GPUs on port $MASTER_PORT"

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
    pip install -U -r "$REQUIREMENTS_FILE"
    
    # Make sure torch distributed is installed
    pip install torch>=2.0.0
else
    echo "No requirements.txt found. Skipping dependency installation."
fi

# Kill any existing processes using the default PyTorch distributed ports
echo "Ensuring no conflicting processes are running..."
if command -v lsof &> /dev/null; then
    # Try to find and kill processes using ports in the 29500-29510 range (common PyTorch ports)
    for port in $(seq 29500 29510); do
        pid=$(lsof -ti:$port 2>/dev/null)
        if [ ! -z "$pid" ]; then
            echo "Killing process $pid using port $port"
            kill -9 $pid 2>/dev/null || true
        fi
    done
fi

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"

# Create results directory if it doesn't exist
mkdir -p results

# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=$MASTER_PORT

# Use torchrun (preferred) if available, otherwise fallback to torch.distributed.launch
if command -v torchrun &> /dev/null; then
    echo "Using torchrun for distributed training..."
    PYTHONPATH=$(pwd) torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_addr=localhost --master_port=$MASTER_PORT src/main_distributed.py
else
    echo "Using torch.distributed.launch for distributed training..."
    PYTHONPATH=$(pwd) python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_addr=localhost --master_port=$MASTER_PORT src/main_distributed.py
fi

echo "Training complete!"
