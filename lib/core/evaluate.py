from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from sklearn.metrics import f1_score
import numpy as np


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if isinstance(output, list):
        output = output[-1]

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


@torch.no_grad()
def f1_metric(output, target, average='macro'):
    """
    Computes F1 score for classification tasks
    
    Args:
        output: Model predictions (logits) - tensor of shape [batch_size, num_classes]
        target: Ground truth labels - tensor of shape [batch_size]
        average: Type of averaging for F1 calculation
                - 'macro': Calculate metrics for each label, and find their unweighted mean
                - 'weighted': Calculate metrics for each label, and find their average weighted by support
                - 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives
    
    Returns:
        F1 score (float): F1 score value (0-1 range, multiplied by 100 for percentage)
    """
    if isinstance(output, list):
        output = output[-1]
    
    # Get predictions from logits
    _, pred = torch.max(output, 1)
    
    # Convert to numpy for sklearn
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # Calculate F1 score
    try:
        f1 = f1_score(target_np, pred_np, average=average, zero_division=0)
        return f1 * 100.0  # Convert to percentage like accuracy
    except Exception as e:
        # Fallback to accuracy if F1 calculation fails
        correct = (pred == target).float().sum()
        total = target.size(0)
        return (correct / total * 100.0).item()


@torch.no_grad()
def f1_and_accuracy(output, target, topk=(1,), f1_average='macro'):
    """
    Computes both F1 score and accuracy for comprehensive evaluation
    
    Args:
        output: Model predictions (logits)
        target: Ground truth labels  
        topk: Top-k accuracy values to compute
        f1_average: Averaging method for F1 score
    
    Returns:
        tuple: (f1_score, accuracy_results)
               f1_score: F1 score as percentage
               accuracy_results: List of top-k accuracy values
    """
    # Calculate F1 score
    f1 = f1_metric(output, target, average=f1_average)
    
    # Calculate accuracy (keep original function for compatibility)
    acc_results = accuracy(output, target, topk=topk)
    
    return f1, acc_results
