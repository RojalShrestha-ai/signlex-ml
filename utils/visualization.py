import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import ASL_LABELS_INV


def get_attention_maps(
    model: torch.nn.Module,
    image: torch.Tensor,
    layer_idx: int = -1,
) -> torch.Tensor:

    model.eval()
    attention_maps = []
    
    # Hook to capture attention weights
    def hook_fn(module, input, output):
        # Capture attention weights before softmax
        attention_maps.append(output.detach())
    
    # Register hooks on attention layers
    hooks = []
    
    if hasattr(model, 'backbone'):
        # Pretrained model wrapper
        blocks = model.backbone.blocks
    elif hasattr(model, 'encoder_layers'):
        # Our custom implementation
        blocks = model.encoder_layers
    else:
        raise ValueError("Unsupported model architecture")
    
    for block in blocks:
        if hasattr(block, 'attn'):
            hooks.append(block.attn.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        _ = model(image)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Get requested layer
    if attention_maps:
        attn = attention_maps[layer_idx]
        return attn[0]  # Remove batch dimension
    else:
        raise ValueError("No attention maps captured")


def visualize_attention(
    image: torch.Tensor,
    attention: torch.Tensor,
    head_idx: Optional[int] = None,
    patch_size: int = 16,
    save_path: Optional[str] = None,
) -> None:

    import matplotlib.pyplot as plt
    
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    
    H, W = img.shape[:2]
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    # Get attention from [CLS] token to all patches
    # attention shape: [num_heads, N, N] where N = num_patches + 1
    cls_attention = attention[:, 0, 1:]  # [num_heads, num_patches]
    
    if head_idx is not None:
        attn_map = cls_attention[head_idx]  # [num_patches]
    else:
        attn_map = cls_attention.mean(dim=0)  # Average over heads
    
    # Reshape to 2D grid
    attn_map = attn_map.reshape(num_patches_h, num_patches_w).cpu().numpy()
    
    # Upscale to image size
    attn_map = np.kron(attn_map, np.ones((patch_size, patch_size)))
    
    # Normalize
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map
    axes[1].imshow(attn_map, cmap='hot')
    axes[1].set_title(f'Attention Map (Head {head_idx if head_idx else "avg"})')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(attn_map, cmap='hot', alpha=0.5)
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    
    return fig


def visualize_all_heads(
    image: torch.Tensor,
    attention: torch.Tensor,
    patch_size: int = 16,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize attention from all heads in a grid.
    
    Args:
        image: Original image [3, H, W]
        attention: Attention weights [num_heads, N, N]
        patch_size: Patch size
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    num_heads = attention.shape[0]
    
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    
    H, W = img.shape[:2]
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    # Create grid
    cols = 4
    rows = (num_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    
    for i in range(num_heads):
        # Get attention from [CLS] to patches
        cls_attention = attention[i, 0, 1:]
        attn_map = cls_attention.reshape(num_patches_h, num_patches_w).cpu().numpy()
        attn_map = np.kron(attn_map, np.ones((patch_size, patch_size)))
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        axes[i].imshow(img)
        axes[i].imshow(attn_map, cmap='hot', alpha=0.5)
        axes[i].set_title(f'Head {i}')
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Attention Maps from All Heads', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved all heads visualization to {save_path}")
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
) -> None:
 
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    return fig


def visualize_predictions(
    images: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_samples: int = 16,
    save_path: Optional[str] = None,
) -> None:

    import matplotlib.pyplot as plt
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs = images.cpu() * std + mean
    imgs = imgs.clamp(0, 1)
    
    # Get predictions
    probs = F.softmax(predictions, dim=1)
    preds = predictions.argmax(dim=1)
    
    # Select samples
    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = imgs[i].permute(1, 2, 0).numpy()
        pred = preds[i].item()
        label = labels[i].item()
        conf = probs[i, pred].item()
        
        axes[i].imshow(img)
        
        color = 'green' if pred == label else 'red'
        title = f"Pred: {ASL_LABELS_INV[pred]} ({conf:.2%})\nTrue: {ASL_LABELS_INV[label]}"
        axes[i].set_title(title, color=color, fontsize=10)
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Model Predictions', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved predictions visualization to {save_path}")
    
    return fig


def visualize_embeddings(
    features: torch.Tensor,
    labels: torch.Tensor,
    method: str = 'tsne',
    save_path: Optional[str] = None,
) -> None:

    import matplotlib.pyplot as plt
    
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Reduce dimensionality
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Running {method.upper()}...")
    embeddings = reducer.fit_transform(features_np)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by class
    num_classes = len(np.unique(labels_np))
    cmap = plt.cm.get_cmap('tab20', num_classes)
    
    for i in range(num_classes):
        mask = labels_np == i
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[cmap(i)],
            label=ASL_LABELS_INV[i],
            alpha=0.6,
            s=20,
        )
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_title(f'{method.upper()} Visualization of ViT Features', fontsize=14)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved embedding visualization to {save_path}")
    
    return fig


def create_gradcam_visualization(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: Optional[int] = None,
    save_path: Optional[str] = None,
) -> None:

    import matplotlib.pyplot as plt
    
    model.eval()
    
    # Enable gradients
    image = image.requires_grad_(True)
    
    # Forward pass
    output = model(image)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass for target class
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    output.backward(gradient=one_hot, retain_graph=True)
    
    # Get gradients w.r.t. input
    gradients = image.grad.data[0]  # [3, H, W]
    
    # Average gradients across channels
    saliency = gradients.abs().mean(dim=0).cpu().numpy()  # [H, W]
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image[0].detach().cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(saliency, cmap='hot')
    axes[1].set_title(f'Saliency Map (Class: {ASL_LABELS_INV[target_class]})')
    axes[1].axis('off')
    
    axes[2].imshow(img)
    axes[2].imshow(saliency, cmap='hot', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved GradCAM visualization to {save_path}")
    
    return fig


if __name__ == "__main__":
    print("Visualization module loaded!")
    print("\nAvailable functions:")
    print("  - get_attention_maps(model, image)")
    print("  - visualize_attention(image, attention)")
    print("  - visualize_all_heads(image, attention)")
    print("  - plot_training_curves(losses, accs)")
    print("  - visualize_predictions(images, preds, labels)")
    print("  - visualize_embeddings(features, labels)")
    print("  - create_gradcam_visualization(model, image)")
