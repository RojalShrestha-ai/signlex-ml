import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    import timm
    from timm.models.vision_transformer import VisionTransformer as TimmViT
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. Run: pip install timm")


class PretrainedViT(nn.Module):

    
    def __init__(
        self,
        model_name: str = "vit_large_patch16_224.augreg_in21k_ft_in1k",
        num_classes: int = 26,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm library required. Install with: pip install timm")
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained model from timm
        # drop_rate: dropout in attention/mlp
        # drop_path_rate: stochastic depth
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        # Get embedding dimension from backbone
        self.embed_dim = self.backbone.embed_dim  # 1024 for ViT-Large
        
        # Create new classification head for ASL
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.5),
            nn.Linear(self.embed_dim, num_classes),
        )
        
        # Optionally freeze backbone for transfer learning
        if freeze_backbone:
            self._freeze_backbone()
            
        # Initialize new head
        self._init_head()
        
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Backbone frozen: {sum(p.numel() for p in self.backbone.parameters()):,} parameters")
        
    def _unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")
        
    def _init_head(self):
        """Initialize classification head with small weights."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x) 
        
        if hasattr(self.backbone, 'fc_norm'):
            features = self.backbone.fc_norm(features[:, 0])
        else:
            features = features[:, 0]  
        
        logits = self.head(features)  
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        if len(features.shape) == 3:
            features = features[:, 0]  # CLS token
        return features
    
    def get_attention_weights(self, x: torch.Tensor) -> list:
        attention_weights = []
        
        # Hook to capture attention
        def hook_fn(module, input, output):
            # For timm ViT, attention is in output
            if hasattr(module, 'attn'):
                attention_weights.append(output)
        
        # Register hooks
        hooks = []
        for block in self.backbone.blocks:
            hooks.append(block.attn.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            _ = self.backbone.forward_features(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights


def create_pretrained_vit(
    model_size: str = "large",
    num_classes: int = 26,
    pretrained: bool = True,
    **kwargs
) -> PretrainedViT:

    model_mapping = {
        "base": "vit_base_patch16_224.augreg_in21k_ft_in1k",
        "large": "vit_large_patch16_224.augreg_in21k_ft_in1k",
        "huge": "vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k",
    }
    
    if model_size not in model_mapping:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(model_mapping.keys())}")
    
    model_name = model_mapping[model_size]
    
    return PretrainedViT(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )


def list_available_vit_models() -> list:
    """List all available ViT models in timm."""
    if not TIMM_AVAILABLE:
        return []
    return timm.list_models("vit_*", pretrained=True)


def model_info(model: PretrainedViT):
    """Print model information."""
    print("=" * 60)
    print("PRETRAINED VIT MODEL INFO")
    print("=" * 60)
    print(f"Model:           {model.model_name}")
    print(f"Embed dim:       {model.embed_dim}")
    print(f"Num classes:     {model.num_classes}")
    
    # Count parameters
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = backbone_params + head_params
    
    print(f"\nParameters:")
    print(f"  Backbone:      {backbone_params:,}")
    print(f"  Head:          {head_params:,}")
    print(f"  Total:         {total:,} ({total/1e6:.1f}M)")
    print(f"  Trainable:     {trainable:,}")
    
    # Memory estimate
    print(f"\nMemory (FP32):   {total * 4 / 1e9:.2f} GB")
    print(f"Memory (BF16):   {total * 2 / 1e9:.2f} GB")
    print("=" * 60)


class ViTWithMixup(PretrainedViT):

    
    def __init__(
        self,
        *args,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        
    def mixup_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:

        lam = torch.distributions.Beta(
            self.mixup_alpha, self.mixup_alpha
        ).sample().item()
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def cutmix_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
     
        lam = torch.distributions.Beta(
            self.cutmix_alpha, self.cutmix_alpha
        ).sample().item()
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        # Get random box
        W, H = x.size(2), x.size(3)
        cut_ratio = (1 - lam) ** 0.5
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        cx = torch.randint(W, (1,)).item()
        cy = torch.randint(H, (1,)).item()
        
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


if __name__ == "__main__":
    if not TIMM_AVAILABLE:
        print("Please install timm: pip install timm")
    else:
        # List some available models
        print("Available ViT models (sample):")
        models = list_available_vit_models()
        for m in models[:10]:
            print(f"  - {m}")
        print(f"  ... and {len(models) - 10} more")
        
        # Create and test model
        print("\n" + "=" * 60)
        print("Creating ViT-Large pretrained model...")
        print("=" * 60)
        
        model = create_pretrained_vit(
            model_size="large",
            num_classes=26,
            pretrained=True,
            drop_path_rate=0.2,
        )
        
        model_info(model)
        
        # Test forward pass
        print("\nTesting forward pass...")
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            y = model(x)
        print(f"Input:  {x.shape}")
        print(f"Output: {y.shape}")
        print(f"Logits: {y[0, :5]}")
