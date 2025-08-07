#!/usr/bin/env python3
"""
Test script untuk memverifikasi config YAML
"""

import sys
import os
sys.path.append('/content/CvT/lib')

from config import config, update_config
import argparse

def test_config():
    """Test loading config file"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', 
                       default='experiments/imagenet/cvt/cvt-21-224x224_paddy_dataset.yaml',
                       type=str)
    args = parser.parse_args()
    
    try:
        print("🧪 Testing config loading...")
        print(f"Config file: {args.cfg}")
        
        # Try to update config
        update_config(config, args)
        
        print("✅ Config loaded successfully!")
        print(f"Dataset root: {config.DATASET.ROOT}")
        print(f"Train set: {config.DATASET.TRAIN_SET}")
        print(f"Val set: {config.DATASET.VAL_SET}")
        print(f"Test set: {config.DATASET.TEST_SET}")
        print(f"Number of classes: {config.MODEL.NUM_CLASSES}")
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False
        
    return True

if __name__ == '__main__':
    success = test_config()
    sys.exit(0 if success else 1)
