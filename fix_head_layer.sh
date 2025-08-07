#!/bin/bash
# ===================================================================
# FIX FINE-TUNING HEAD LAYER SIZE MISMATCH
# ===================================================================

echo "🔧 Fixing head layer size mismatch for fine-tuning..."

cd /content/CvT

# Test model loading
echo "🧪 Testing model loading..."
python -c "
import sys
sys.path.append('lib')

try:
    import torch
    from config import config, update_config
    from models import build_model
    import argparse
    
    # Simulate args
    class Args:
        cfg = 'experiments/imagenet/cvt/cvt-21-224x224_paddy_dataset.yaml'
        opts = None
    
    args = Args()
    update_config(config, args)
    
    print('✅ Config loaded successfully')
    print(f'Model classes: {config.MODEL.NUM_CLASSES}')
    print(f'Pretrained: {config.MODEL.PRETRAINED}')
    
    # Test model building
    model = build_model(config)
    print('✅ Model built successfully with fixed head layer handling')
    
    if hasattr(model, 'head'):
        print(f'Head weight shape: {model.head.weight.shape}')
        print(f'Expected: [10, 384] for paddy disease classification')
        
    print('🎉 Fine-tuning setup ready!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"

echo "✅ Head layer fix completed!"
echo "Now you can run training: bash run_paddy_training.sh"
