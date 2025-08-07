#!/usr/bin/env python3
"""
Test model loading dengan pretrained weights untuk fine-tuning
"""

import sys
import os
sys.path.append('/content/CvT/lib')

import torch
from config import config, update_config
from models import build_model
import argparse

def test_model_loading():
    """Test loading model dengan pretrained weights"""
    
    # Setup config
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', 
                       default='experiments/imagenet/cvt/cvt-21-224x224_paddy_dataset.yaml',
                       type=str)
    args = parser.parse_args()
    
    try:
        print("🧪 Testing model loading with pretrained weights...")
        
        # Update config
        update_config(config, args)
        print(f"✅ Config loaded")
        print(f"Model classes: {config.MODEL.NUM_CLASSES}")
        print(f"Pretrained path: {config.MODEL.PRETRAINED}")
        
        # Build model
        model = build_model(config)
        print(f"✅ Model built successfully")
        
        # Check model head
        if hasattr(model, 'head'):
            print(f"Head layer shape: {model.head.weight.shape}")
            print(f"Expected shape: [10, 384] for 10 classes")
        
        print("🎉 Model loading test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_model_loading()
    sys.exit(0 if success else 1)
