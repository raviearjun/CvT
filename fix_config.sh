#!/bin/bash
# ===================================================================
# FIX KONFIGURASI CVT UNTUK GOOGLE COLAB
# ===================================================================

echo "🔧 Fixing CvT configuration for Google Colab..."

# 1. Test if config can be loaded
echo "🧪 Testing config loading..."
cd /content/CvT

python -c "
import sys
sys.path.append('lib')
try:
    from config import config
    print('✅ Base config loaded successfully')
    print(f'Available dataset keys: {list(config.DATASET.keys())}')
except Exception as e:
    print(f'❌ Base config failed: {e}')
"

# 2. Test YAML config loading
echo "🧪 Testing YAML config loading..."
python -c "
import sys
sys.path.append('lib')
try:
    from config import config, update_config
    import argparse
    
    # Simulate args
    class Args:
        cfg = 'experiments/imagenet/cvt/cvt-21-224x224_paddy_dataset.yaml'
        opts = None
    
    args = Args()
    update_config(config, args)
    print('✅ YAML config loaded successfully')
    print(f'Train set: {config.DATASET.TRAIN_SET}')
    print(f'Val set: {config.DATASET.VAL_SET}')
    print(f'Test set: {config.DATASET.TEST_SET}')
    
except Exception as e:
    print(f'❌ YAML config failed: {e}')
    import traceback
    traceback.print_exc()
"

echo "✅ Configuration test completed!"
echo "Now try running: bash run_paddy_training.sh"
