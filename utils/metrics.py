import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import ASL_LABELS, ASL_LABELS_INV


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5)
) -> Dict[str, float]:

    maxk = max(topk)
    batch_size = labels.size(0)
    
    # Get top-k predictions
    _, pred = predictions.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # [maxk, N]
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    
    result = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        result[f'top{k}_acc'] = (correct_k / batch_size).item()
    
    return result


def compute_per_class_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 26
) -> Dict[int, float]:
    """
    Compute accuracy for each class.
    
    Returns:
        Dictionary mapping class index to accuracy
    """
    preds = predictions.argmax(dim=1) if predictions.dim() > 1 else predictions
    
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    
    for pred, label in zip(preds.tolist(), labels.tolist()):
        per_class_total[label] += 1
        if pred == label:
            per_class_correct[label] += 1
    
    per_class_acc = {}
    for cls in range(num_classes):
        if per_class_total[cls] > 0:
            per_class_acc[cls] = per_class_correct[cls] / per_class_total[cls]
        else:
            per_class_acc[cls] = 0.0
    
    return per_class_acc


def compute_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 26
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Returns:
        np.ndarray of shape [num_classes, num_classes]
        Row = true label, Column = predicted label
    """
    preds = predictions.argmax(dim=1) if predictions.dim() > 1 else predictions
    
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for pred, label in zip(preds.tolist(), labels.tolist()):
        confusion[label, pred] += 1
    
    return confusion


def compute_precision_recall_f1(
    confusion_matrix: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute precision, recall, and F1 score from confusion matrix.
    
    Returns:
        Dictionary with precision, recall, f1 arrays (one per class)
    """
    num_classes = confusion_matrix.shape[0]
    
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1[i] = 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': precision.mean(),
        'macro_recall': recall.mean(),
        'macro_f1': f1.mean(),
    }


def compute_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 26
) -> Dict:

    metrics = {}
    
    # Top-k accuracy
    topk_acc = compute_accuracy(predictions, labels, topk=(1, 5))
    metrics.update(topk_acc)
    
    # Per-class accuracy
    per_class_acc = compute_per_class_accuracy(predictions, labels, num_classes)
    metrics['per_class_accuracy'] = per_class_acc
    metrics['mean_class_accuracy'] = np.mean(list(per_class_acc.values()))
    
    # Confusion matrix
    confusion = compute_confusion_matrix(predictions, labels, num_classes)
    metrics['confusion_matrix'] = confusion
    
    # Precision, recall, F1
    prf = compute_precision_recall_f1(confusion)
    metrics.update(prf)
    
    return metrics


def print_classification_report(
    metrics: Dict,
    class_names: Optional[List[str]] = None
):
    """Print a formatted classification report."""
    if class_names is None:
        class_names = [ASL_LABELS_INV[i] for i in range(26)]
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    
    print(f"\n{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    print("-"*50)
    
    for i, name in enumerate(class_names):
        print(f"{name:<10} "
              f"{metrics['precision'][i]:>10.4f} "
              f"{metrics['recall'][i]:>10.4f} "
              f"{metrics['f1'][i]:>10.4f} "
              f"{metrics['per_class_accuracy'][i]:>10.4f}")
    
    print("-"*50)
    print(f"{'Macro Avg':<10} "
          f"{metrics['macro_precision']:>10.4f} "
          f"{metrics['macro_recall']:>10.4f} "
          f"{metrics['macro_f1']:>10.4f} "
          f"{metrics['mean_class_accuracy']:>10.4f}")
    
    print(f"\nTop-1 Accuracy: {metrics['top1_acc']:.4f}")
    print(f"Top-5 Accuracy: {metrics['top5_acc']:.4f}")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    normalize: bool = True,
):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        confusion_matrix: [num_classes, num_classes] array
        class_names: List of class names
        save_path: Path to save figure (optional)
        figsize: Figure size
        normalize: Normalize by row (true class)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if class_names is None:
        class_names = [ASL_LABELS_INV[i] for i in range(confusion_matrix.shape[0])]
    
    # Normalize
    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        cm = confusion_matrix.astype(float) / row_sums
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        cm = confusion_matrix
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    return fig


def find_misclassified(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    image_paths: Optional[List[str]] = None,
    num_samples: int = 10,
) -> List[Dict]:
    """
    Find misclassified samples for error analysis.
    
    Returns:
        List of dicts with misclassification info
    """
    preds = predictions.argmax(dim=1)
    probs = torch.softmax(predictions, dim=1)
    
    misclassified = []
    
    for i, (pred, label, prob) in enumerate(zip(preds, labels, probs)):
        if pred != label:
            info = {
                'index': i,
                'true_label': ASL_LABELS_INV[label.item()],
                'pred_label': ASL_LABELS_INV[pred.item()],
                'confidence': prob[pred].item(),
                'true_prob': prob[label].item(),
            }
            if image_paths:
                info['path'] = image_paths[i]
            misclassified.append(info)
    
    # Sort by confidence (high confidence errors are most concerning)
    misclassified.sort(key=lambda x: -x['confidence'])
    
    return misclassified[:num_samples]


def compute_error_analysis(
    confusion_matrix: np.ndarray,
    top_n: int = 5,
) -> Dict:
    """
    Analyze most common errors.
    
    Returns:
        Dictionary with error analysis
    """
    num_classes = confusion_matrix.shape[0]
    
    # Find top confusions (excluding diagonal)
    errors = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion_matrix[i, j] > 0:
                errors.append({
                    'true': ASL_LABELS_INV[i],
                    'pred': ASL_LABELS_INV[j],
                    'count': int(confusion_matrix[i, j]),
                })
    
    errors.sort(key=lambda x: -x['count'])
    
    return {
        'top_errors': errors[:top_n],
        'total_errors': sum(e['count'] for e in errors),
        'error_rate': sum(e['count'] for e in errors) / confusion_matrix.sum(),
    }


class MetricsTracker:

    
    def __init__(self, num_classes: int = 26):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.all_predictions = []
        self.all_labels = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        """Add batch of predictions and labels."""
        self.all_predictions.append(predictions.detach().cpu())
        self.all_labels.append(labels.detach().cpu())
    
    def compute(self) -> Dict:
        """Compute all metrics from accumulated predictions."""
        predictions = torch.cat(self.all_predictions, dim=0)
        labels = torch.cat(self.all_labels, dim=0)
        return compute_metrics(predictions, labels, self.num_classes)


if __name__ == "__main__":
    print("Testing metrics module...")
    
    num_samples = 1000
    num_classes = 26
    
    predictions = torch.randn(num_samples, num_classes)
    
    # Random labels
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Compute metrics
    metrics = compute_metrics(predictions, labels, num_classes)
    
    # Print report
    print_classification_report(metrics)
    
    # Error analysis
    errors = compute_error_analysis(metrics['confusion_matrix'])
    print(f"\nTop 5 confusions:")
    for e in errors['top_errors']:
        print(f"  {e['true']} → {e['pred']}: {e['count']}")
    
    # Plot confusion matrix
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        fig = plot_confusion_matrix(
            metrics['confusion_matrix'],
            save_path='test_confusion_matrix.png'
        )
        print("\nConfusion matrix plot saved!")
    except ImportError:
        print("\nMatplotlib not available for plotting")
    
    print("\nMetrics module test complete!")
