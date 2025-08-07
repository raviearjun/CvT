#!/usr/bin/env python3
"""
Setup script untuk membantu persiapan fine-tuning CvT di Google Colab
"""

import os
import shutil
import argparse
from pathlib import Path

def setup_directories():
    """Buat direktori yang diperlukan"""
    directories = [
        '/content/CvT/paddy_disease_classification/train',
        '/content/CvT/paddy_disease_classification/val', 
        '/content/output'
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def copy_from_drive(drive_dataset_path, drive_weights_path):
    """Copy dataset dan weights dari Google Drive"""
    
    # Copy dataset
    if os.path.exists(drive_dataset_path):
        print(f"Copying dataset from {drive_dataset_path}...")
        shutil.copytree(drive_dataset_path, '/content/CvT/paddy_disease_classification', dirs_exist_ok=True)
        print("✓ Dataset copied successfully")
    else:
        print(f"❌ Dataset not found at {drive_dataset_path}")
    
    # Copy weights
    if os.path.exists(drive_weights_path):
        weights_dest = '/content/CvT/CvT-21-224x224-IN-1k.pth'
        shutil.copy2(drive_weights_path, weights_dest)
        print(f"✓ Weights copied to {weights_dest}")
    else:
        print(f"❌ Weights not found at {drive_weights_path}")

def verify_dataset_structure():
    """Verifikasi struktur dataset"""
    dataset_root = '/content/CvT/paddy_disease_classification'
    
    for split in ['train', 'val']:
        split_path = os.path.join(dataset_root, split)
        if not os.path.exists(split_path):
            print(f"❌ Missing {split} directory")
            continue
            
        classes = [d for d in os.listdir(split_path) 
                  if os.path.isdir(os.path.join(split_path, d))]
        
        print(f"\n{split.upper()} dataset:")
        print(f"  Classes found: {len(classes)}")
        
        total_images = 0
        for class_name in sorted(classes):
            class_path = os.path.join(split_path, class_name)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"  {class_name}: {len(images)} images")
            total_images += len(images)
        
        print(f"  Total {split} images: {total_images}")

def check_requirements():
    """Check apakah semua requirements sudah terinstall"""
    import importlib
    
    required_packages = [
        'torch', 'torchvision', 'timm', 'yacs', 
        'tensorboard', 'cv2', 'PIL', 'numpy', 
        'matplotlib', 'sklearn', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                importlib.import_module('cv2')
            elif package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Run: !pip install -r requirements.txt")
    else:
        print("\n✓ All required packages are installed")

def main():
    parser = argparse.ArgumentParser(description='Setup CvT fine-tuning environment')
    parser.add_argument('--dataset-path', type=str, 
                       default='/content/drive/MyDrive/paddy_dataset',
                       help='Path to dataset in Google Drive')
    parser.add_argument('--weights-path', type=str,
                       default='/content/drive/MyDrive/CvT-21-224x224-IN-1k.pth',
                       help='Path to pretrained weights in Google Drive')
    parser.add_argument('--skip-copy', action='store_true',
                       help='Skip copying files from Drive')
    
    args = parser.parse_args()
    
    print("🚀 Setting up CvT fine-tuning environment...")
    print("=" * 50)
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Copy files from Drive
    if not args.skip_copy:
        print("\n📋 Copying files from Google Drive...")
        copy_from_drive(args.dataset_path, args.weights_path)
    
    # Verify dataset
    print("\n🔍 Verifying dataset structure...")
    verify_dataset_structure()
    
    # Check requirements
    print("\n📦 Checking requirements...")
    check_requirements()
    
    print("\n✅ Setup completed!")
    print("\nNext steps:")
    print("1. Verify dataset and weights are correctly placed")
    print("2. Run: python tools/train.py --cfg experiments/imagenet/cvt/cvt-21-224x224.yaml")

if __name__ == '__main__':
    main()
