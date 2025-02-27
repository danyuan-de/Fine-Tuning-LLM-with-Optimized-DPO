# Make the script executable with the following command:
# chmod +x setup.sh
# Run the script with the following command:
# ./setup.sh


#!/bin/bash

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

# Check if requirements.txt exists and install dependencies
REQUIREMENTS_FILE="requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "No requirements.txt found. Skipping dependency installation."
fi

# Run dpoTraining.py if it exists
SCRIPT_FILE="dpoTraining.py"
if [ -f "$SCRIPT_FILE" ]; then
    echo "Running $SCRIPT_FILE..."
    python "$SCRIPT_FILE"
else
    echo "Error: $SCRIPT_FILE not found!"
    exit 1
fi
