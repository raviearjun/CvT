#!/bin/bash
# ===================================================================
# QUICK FIX FOR CVT COLAB TRAINING ERRORS
# Run this script in Google Colab to fix all dependency issues
# ===================================================================

echo "🔧 Installing missing dependencies..."

# Install missing tensorwatch and other packages
pip install tensorwatch yacs tensorboardX timm einops json_tricks ptflops --quiet

echo "✅ Dependencies installed!"

# Test essential imports
echo "🧪 Testing imports..."
python -c "
try:
    import torch
    import torchvision
    import timm
    import yacs
    import tensorboardX
    import tensorwatch
    import einops
    print('✅ All imports successful!')
    print(f'PyTorch: {torch.__version__}')
    print(f'TorchVision: {torchvision.__version__}')
    print(f'TIMM: {timm.__version__}')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

echo "🚀 Ready to run training!"
echo "Now you can run: bash run_paddy_training.sh"
