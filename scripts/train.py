#!/usr/bin/env python3
import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config, get_config, ASL_LABELS_INV
from data.dataset import ASLDataset, get_train_transforms, get_val_transforms
from models.vit_pretrained import create_pretrained_vit, model_info


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Cosine learning rate schedule with linear warmup.
    
    Warmup: Linear increase from 0 to max_lr
    Decay: Cosine decay from max_lr to min_lr
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp data augmentation.
    
    Blends two random samples: x' = λx_i + (1-λ)x_j
    Loss is computed as: λ*CE(y_i) + (1-λ)*CE(y_j)
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix data augmentation.
    
    Cuts and pastes a rectangular region between two samples.
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    # Get random box
    W, H = x.size(2), x.size(3)
    cut_ratio = (1 - lam) ** 0.5
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    
    # Random center
    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    
    # Box coordinates
    bbx1 = max(0, cx - cut_w // 2)
    bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(W, cx + cut_w // 2)
    bby2 = min(H, cy + cut_h // 2)
    
    # Apply cut
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda based on actual area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixed loss for MixUp/CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model on validation/test set.
    
    Returns:
        loss: Average loss
        accuracy: Top-1 accuracy
    """
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        with autocast(dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Calculate accuracy
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc.item(), images.size(0))
    
    return loss_meter.avg, acc_meter.avg


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: Config,
    writer: Optional[SummaryWriter] = None,
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        loss: Average training loss
        accuracy: Training accuracy
    """
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Apply MixUp or CutMix with 50% probability
        use_mixup = torch.rand(1).item() < config.augmentation.mixup_cutmix_prob
        
        if use_mixup and torch.rand(1).item() < 0.5:
            images, labels_a, labels_b, lam = mixup_data(
                images, labels, config.augmentation.mixup_alpha
            )
        elif use_mixup:
            images, labels_a, labels_b, lam = cutmix_data(
                images, labels, config.augmentation.cutmix_alpha
            )
        else:
            labels_a, labels_b, lam = labels, labels, 1.0
        
        # Forward pass with mixed precision
        with autocast(dtype=torch.bfloat16):
            outputs = model(images)
            
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            config.training.gradient_clip_max_norm
        )
        
        # Update weights
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Calculate accuracy (use original labels for accuracy)
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc.item(), images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
        
        # TensorBoard logging
        if writer and batch_idx % config.training.log_interval == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss_step', loss.item(), global_step)
            writer.add_scalar('train/acc_step', acc.item(), global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
    
    return loss_meter.avg, acc_meter.avg


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    best_acc: float,
    config: Config,
    path: str,
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_acc': best_acc,
        'config': config.to_dict(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train ViT for ASL recognition")
    
    # Data
    parser.add_argument('--train_manifest', type=str, default='./data/processed/train.json')
    parser.add_argument('--val_manifest', type=str, default='./data/processed/val.json')
    parser.add_argument('--test_manifest', type=str, default='./data/processed/test.json')
    
    # Model
    parser.add_argument('--model_size', type=str, default='large', choices=['base', 'large', 'huge'])
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--drop_path_rate', type=float, default=0.2)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    
    # System
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Logging
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load config
    config = get_config()
    config.data.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.weight_decay = args.weight_decay
    config.training.warmup_epochs = args.warmup_epochs
    config.training.label_smoothing = args.label_smoothing
    config.data.num_workers = args.num_workers
    config.model.drop_path_rate = args.drop_path_rate
    
    # Setup output directories
    exp_name = args.exp_name or f"vit_{args.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / exp_name
    checkpoint_dir = output_dir / 'checkpoints'
    tensorboard_dir = output_dir / 'tensorboard'
    
    for d in [output_dir, checkpoint_dir, tensorboard_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SIGNLEX VIT-LARGE TRAINING")
    print(f"{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Output: {output_dir}")
    
    # Create datasets
    print(f"\nLoading datasets...")
    
    train_transform = get_train_transforms(config.data.image_size)
    val_transform = get_val_transforms(config.data.image_size)
    
    train_dataset = ASLDataset(args.train_manifest, transform=train_transform)
    val_dataset = ASLDataset(args.val_manifest, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    
    # Create model
    print(f"\nCreating model: ViT-{args.model_size.upper()}...")
    
    model = create_pretrained_vit(
        model_size=args.model_size,
        num_classes=26,
        pretrained=args.pretrained,
        drop_path_rate=args.drop_path_rate,
    )
    model = model.to(device)
    model_info(model)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=config.training.min_learning_rate / args.lr,
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # TensorBoard writer
    writer = SummaryWriter(tensorboard_dir)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        min_delta=config.training.early_stopping_min_delta,
        mode='max'
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resumed from epoch {start_epoch}, best_acc: {best_acc:.4f}")
    
    # Training loop
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"{'='*60}\n")
    
    train_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            config=config,
            writer=writer,
        )
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start_time
        
        # Log to TensorBoard
        writer.add_scalar('train/loss_epoch', train_loss, epoch)
        writer.add_scalar('train/acc_epoch', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs-1} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Save best model
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, best_acc, config,
                str(checkpoint_dir / 'best.pth')
            )
            print(f"  ✓ New best! Saved to {checkpoint_dir / 'best.pth'}")
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch, best_acc, config,
            str(checkpoint_dir / 'last.pth')
        )
        
        # Early stopping
        if early_stopping(val_acc):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % config.training.save_every_n_epochs == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, best_acc, config,
                str(checkpoint_dir / f'epoch_{epoch}.pth')
            )
    
    # Training complete
    total_time = time.time() - train_start_time
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Best model saved to: {checkpoint_dir / 'best.pth'}")
    
    # Final evaluation on test set
    if Path(args.test_manifest).exists():
        print(f"\n{'='*60}")
        print("EVALUATING ON TEST SET")
        print(f"{'='*60}")
        
        # Load best model
        load_checkpoint(str(checkpoint_dir / 'best.pth'), model)
        
        test_dataset = ASLDataset(args.test_manifest, transform=val_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        writer.add_scalar('test/loss', test_loss, 0)
        writer.add_scalar('test/acc', test_acc, 0)
    
    writer.close()
    print(f"\nTensorBoard logs: {tensorboard_dir}")
    print("Run: tensorboard --logdir outputs/")


if __name__ == "__main__":
    main()
