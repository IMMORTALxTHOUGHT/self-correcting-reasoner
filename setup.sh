#!/bin/bash
set -e

echo "Setting up Self-Correcting Reasoner project..."

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
python -c "from trl import SFTTrainer, GRPOTrainer; print('TRL OK')"
python -c "from math_verify import parse, verify; print('math_verify OK')"

echo "Setup complete. Activate: source venv/bin/activate"
