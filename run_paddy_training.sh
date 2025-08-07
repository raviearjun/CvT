#!/bin/bash

# Script untuk menjalankan fine-tuning paddy disease classification di Google Colab
# Usage: bash run_paddy_training.sh

echo "🌾 Starting CvT Fine-tuning for Paddy Disease Classification..."
echo "=" * 50

# Check if running on Colab
if [ -d "/content" ]; then
    echo "✓ Running on Google Colab"
    WORKING_DIR="/content/CvT"
else
    echo "✓ Running on local machine"
    WORKING_DIR="."
fi

cd $WORKING_DIR

# Install dependencies if needed
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if dataset exists (from manual upload)
echo "🔍 Checking for uploaded dataset..."

# Check if dataset exists (user should populate manually)
echo "🔍 Checking dataset..."

if [ ! -d "$WORKING_DIR/paddy_disease_dataset" ]; then
    echo "❌ Dataset directory not found!"
    echo "   Please create: $WORKING_DIR/paddy_disease_dataset/"
    exit 1
fi

# Check if dataset is populated
train_classes=$(find "$WORKING_DIR/paddy_disease_dataset/train" -maxdepth 1 -type d 2>/dev/null | wc -l)
val_classes=$(find "$WORKING_DIR/paddy_disease_dataset/val" -maxdepth 1 -type d 2>/dev/null | wc -l)

if [ "$train_classes" -le 1 ] || [ "$val_classes" -le 1 ]; then
    echo "❌ Dataset appears to be empty!"
    echo "   Please populate the following directories with your paddy disease images:"
    echo "   - $WORKING_DIR/paddy_disease_dataset/train/"
    echo "   - $WORKING_DIR/paddy_disease_dataset/val/"
    echo ""
    echo "Expected structure:"
    echo "paddy_disease_dataset/"
    echo "├── train/"
    echo "│   ├── bacterial_leaf_blight/"
    echo "│   ├── bacterial_leaf_streak/"
    echo "│   └── ... (other disease classes)"
    echo "└── val/"
    echo "    ├── bacterial_leaf_blight/"
    echo "    ├── bacterial_leaf_streak/"
    echo "    └── ... (other disease classes)"
    exit 1
else
    echo "✅ Dataset found with $((train_classes-1)) train classes and $((val_classes-1)) val classes"
fi

# Check if pretrained weights exist
echo "🔍 Checking pretrained weights..."

if [ ! -f "$WORKING_DIR/CvT-21-224x224-IN-1k.pth" ]; then
    echo "❌ Pretrained weights not found!"
    echo "   Please download and place: CvT-21-224x224-IN-1k.pth in $WORKING_DIR/"
    echo "   You can download from the official CvT model zoo"
    exit 1
else
    echo "✅ Pretrained weights found at $WORKING_DIR/CvT-21-224x224-IN-1k.pth"
fi

# Create output directory
mkdir -p /content/output

# Display dataset information
echo "📊 Dataset Information:"
echo "Train classes: $(ls $WORKING_DIR/paddy_disease_dataset/train/ | wc -l)"
echo "Val classes: $(ls $WORKING_DIR/paddy_disease_dataset/val/ | wc -l)"

for split in train val; do
    echo "\n$split dataset:"
    for class_dir in $WORKING_DIR/paddy_disease_dataset/$split/*/; do
        class_name=$(basename "$class_dir")
        count=$(find "$class_dir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
        echo "  $class_name: $count images"
    done
done

# Start training
echo "\n🚀 Starting training..."
echo "Config: experiments/imagenet/cvt/cvt-21-224x224_paddy_dataset.yaml"
echo "Output: /content/output/"

python tools/train.py \
    --cfg experiments/imagenet/cvt/cvt-21-224x224_paddy_dataset.yaml \
    2>&1 | tee /content/output/training.log

echo "\n✅ Training completed!"
echo "Results saved to: /content/output/"
echo "Files:"
echo "  - best.pth: Best model checkpoint"
echo "  - latest.pth: Latest model checkpoint"
echo "  - log.txt: Detailed training log"
echo "  - training.log: Console output"

# Create downloadable archive
echo "\n� Creating downloadable archive..."
cd /content
zip -r cvt_paddy_results.zip output/
echo "✅ Results archived as: /content/cvt_paddy_results.zip"
echo "   You can download this file from Colab's file browser"

echo "\n🎉 Fine-tuning completed successfully!"
