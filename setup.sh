#!/bin/bash
# Setup script for Self-Correcting Reasoner project

set -e

echo "Setting up Self-Correcting Reasoner project..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Verify Unsloth
python -c "from unsloth import FastLanguageModel; print('Unsloth imported successfully')"

# Verify TRL
python -c "from trl import GRPOTrainer; print('TRL GRPOTrainer imported successfully')"

# Verify math_verify
python -c "from math_verify import parse, verify; print('math_verify imported successfully')"

echo "Setup complete!"
echo "Activate with: source venv/bin/activate"
