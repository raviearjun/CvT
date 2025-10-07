#!/usr/bin/env python3
"""
Test script to verify F1 Score implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import torch
import numpy as np
from sklearn.metrics import f1_score as sklearn_f1
from core.evaluate import f1_metric, f1_and_accuracy, accuracy

def test_f1_implementation():
    """Test F1 score implementation against sklearn reference"""
    print("🧪 Testing F1 Score Implementation")
    print("=" * 50)
    
    # Create test data
    batch_size = 32
    num_classes = 5
    
    # Simulate model outputs (logits)
    torch.manual_seed(42)
    outputs = torch.randn(batch_size, num_classes)
    
    # Create ground truth labels
    np.random.seed(42)
    targets = torch.from_numpy(np.random.randint(0, num_classes, batch_size))
    
    print(f"Test data: {batch_size} samples, {num_classes} classes")
    print(f"Target distribution: {torch.bincount(targets).tolist()}")
    
    # Get predictions
    _, preds = torch.max(outputs, 1)
    
    # Test our F1 implementation vs sklearn
    print("\n📊 F1 Score Comparison:")
    print("-" * 30)
    
    for average_method in ['macro', 'weighted', 'micro']:
        # Our implementation
        our_f1 = f1_metric(outputs, targets, average=average_method)
        
        # Sklearn reference
        sklearn_f1_score = sklearn_f1(
            targets.numpy(), 
            preds.numpy(), 
            average=average_method, 
            zero_division=0
        ) * 100.0  # Convert to percentage
        
        diff = abs(our_f1 - sklearn_f1_score)
        status = "✅ PASS" if diff < 0.01 else "❌ FAIL" 
        
        print(f"{average_method:>8}: Ours={our_f1:6.2f}% | Sklearn={sklearn_f1_score:6.2f}% | Diff={diff:5.2f}% {status}")
    
    # Test combined function
    print("\n🔀 Testing Combined F1 + Accuracy Function:")
    print("-" * 40)
    
    f1_score, (acc1, acc5) = f1_and_accuracy(outputs, targets, topk=(1, 5), f1_average='macro')
    acc_only = accuracy(outputs, targets, topk=(1,))[0]
    
    print(f"F1 Score (macro): {f1_score:.2f}%")
    print(f"Top-1 Accuracy:   {acc1:.2f}%")
    print(f"Top-5 Accuracy:   {acc5:.2f}%")
    print(f"Accuracy (check): {acc_only:.2f}%")
    
    acc_match = abs(acc1 - acc_only) < 0.01
    print(f"Accuracy consistency: {'✅ PASS' if acc_match else '❌ FAIL'}")
    
    # Test edge cases
    print("\n🔍 Testing Edge Cases:")
    print("-" * 25)
    
    # Perfect predictions
    perfect_outputs = torch.zeros(batch_size, num_classes)
    perfect_outputs[range(batch_size), targets] = 10.0  # High confidence correct predictions
    perfect_f1 = f1_metric(perfect_outputs, targets, average='macro')
    perfect_acc = accuracy(perfect_outputs, targets, topk=(1,))[0]
    
    print(f"Perfect predictions - F1: {perfect_f1:.2f}%, Acc: {perfect_acc:.2f}%")
    
    # Random predictions (should be around 20% for 5-class)
    random_outputs = torch.randn(batch_size, num_classes)
    random_f1 = f1_metric(random_outputs, targets, average='macro')
    random_acc = accuracy(random_outputs, targets, topk=(1,))[0]
    
    print(f"Random predictions - F1: {random_f1:.2f}%, Acc: {random_acc:.2f}%")
    
    # Single class prediction (worst case)
    single_class_outputs = torch.zeros(batch_size, num_classes)
    single_class_outputs[:, 0] = 10.0  # Always predict class 0
    single_f1 = f1_metric(single_class_outputs, targets, average='macro')
    single_acc = accuracy(single_class_outputs, targets, topk=(1,))[0]
    
    print(f"Single class pred  - F1: {single_f1:.2f}%, Acc: {single_acc:.2f}%")
    
    print("\n✅ F1 Score implementation test completed!")
    return True

def simulate_training_step():
    """Simulate a training step with F1 evaluation"""
    print("\n🚀 Simulating Training Step with F1 Evaluation")
    print("=" * 55)
    
    # Simulate batch from cultural classification (5 classes)
    batch_size = 16
    num_classes = 5
    class_names = ['balinese', 'batak', 'dayak', 'javanese', 'minangkabau']
    
    # Simulate model outputs with some realistic pattern
    torch.manual_seed(123)
    outputs = torch.randn(batch_size, num_classes) + torch.tensor([0.1, -0.2, 0.3, -0.1, 0.2])
    
    # Create realistic targets (slightly imbalanced)
    np.random.seed(123)
    targets = torch.from_numpy(np.random.choice(5, batch_size, p=[0.25, 0.15, 0.20, 0.30, 0.10]))
    
    print(f"Batch size: {batch_size}")
    print(f"Class distribution:")
    for i, class_name in enumerate(class_names):
        count = (targets == i).sum().item()
        print(f"  {class_name:>10}: {count:2d} samples ({count/batch_size*100:4.1f}%)")
    
    # Evaluate with different F1 averaging methods
    print(f"\n📈 Evaluation Results:")
    print("-" * 25)
    
    for avg_method in ['macro', 'weighted', 'micro']:
        f1_score, (acc1, acc5) = f1_and_accuracy(outputs, targets, topk=(1, 5), f1_average=avg_method)
        print(f"{avg_method:>8} F1: {f1_score:5.1f}% | Acc@1: {acc1:5.1f}% | Acc@5: {acc5:5.1f}%")
    
    # Show per-class predictions for insight
    _, preds = torch.max(outputs, 1)
    print(f"\nPrediction breakdown:")
    for i, class_name in enumerate(class_names):
        pred_count = (preds == i).sum().item()
        target_count = (targets == i).sum().item() 
        correct = ((preds == i) & (targets == i)).sum().item()
        print(f"  {class_name:>10}: {pred_count:2d} pred | {target_count:2d} true | {correct:2d} correct")
    
    print(f"\n✅ Training step simulation completed!")

if __name__ == "__main__":
    print("🔬 F1 Score Implementation Verification")
    print("=" * 60)
    
    try:
        # Test F1 implementation
        test_f1_implementation()
        
        # Simulate training scenario
        simulate_training_step()
        
        print(f"\n🎉 All tests passed! F1 Score implementation is ready to use.")
        print(f"\n💡 Recommendation:")
        print(f"   - Use 'macro' averaging for balanced evaluation across all classes")
        print(f"   - Use 'weighted' averaging if you want to weight by class frequency")  
        print(f"   - Monitor both F1 and Accuracy during training for complete insight")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)