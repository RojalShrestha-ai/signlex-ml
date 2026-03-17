import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from functools import partial


class PatchEmbedding(nn.Module):

    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 1024,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 196 for 224/16
        
        # Conv2D acts as both patch extraction AND linear projection
        # kernel_size = stride = patch_size ensures non-overlapping patches
        self.projection = nn.Conv2d(
            in_channels,      # 3 (RGB)
            embed_dim,        # 1024
            kernel_size=patch_size,  # 16
            stride=patch_size        # 16 (no overlap)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        
        # Apply convolution: [B, 3, 224, 224] → [B, 1024, 14, 14]
        x = self.projection(x)
        
        # Flatten spatial dimensions and transpose: [B, 1024, 14, 14] → [B, 196, 1024]
        x = x.flatten(2)  # [B, 1024, 196]
        x = x.transpose(1, 2)  # [B, 196, 1024]
        
        return x


class MultiHeadSelfAttention(nn.Module):

    
    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 64
        self.scale = self.head_dim ** -0.5  # 1/√64 = 0.125
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        # Q, K, V projections combined for efficiency
        # Instead of 3 separate [1024, 1024] matrices, one [1024, 3072] matrix
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        
        # Output projection after concatenating heads
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(projection_dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
  
        B, N, D = x.shape  # [batch, 197, 1024]
        
        # Step 1: Compute Q, K, V for all heads at once
        # [B, 197, 1024] → [B, 197, 3072] → [B, 197, 3, 16, 64]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, 16, 197, 64]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, 16, 197, 64]
        
        # Step 2: Scaled Dot-Product Attention
        # attention = softmax(Q @ K^T / √d_k) @ V
        
        # Q @ K^T: [B, 16, 197, 64] @ [B, 16, 64, 197] → [B, 16, 197, 197]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Softmax over last dimension (each token's attention over all tokens)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Attention @ V: [B, 16, 197, 197] @ [B, 16, 197, 64] → [B, 16, 197, 64]
        x = attn @ v
        
        # Step 3: Concatenate heads and project
        # [B, 16, 197, 64] → [B, 197, 16, 64] → [B, 197, 1024]
        x = x.transpose(1, 2).reshape(B, N, D)
        
        # Final linear projection
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


class MLP(nn.Module):

    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        out_features = out_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()  # Gaussian Error Linear Unit (smoother than ReLU)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] = [B, 197, 1024]
            
        Returns:
            [B, N, D] = [B, 197, 1024]
        """
        x = self.fc1(x)      # [B, 197, 1024] → [B, 197, 4096]
        x = self.act(x)       # GELU activation
        x = self.dropout(x)
        x = self.fc2(x)      # [B, 197, 4096] → [B, 197, 1024]
        x = self.dropout(x)
        return x


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) Regularization.
    
    Randomly drops entire residual branches during training.
    This helps with regularization and allows training deeper networks.
    
    Different from Dropout: DropPath drops entire samples in a batch,
    while Dropout drops individual neurons.
    
    Paper: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
            
        keep_prob = 1 - self.drop_prob
        # Create random tensor for each sample in batch
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # [B, 1, 1]
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize to 0 or 1
        
        # Scale output to maintain expected value
        output = x.div(keep_prob) * random_tensor
        return output


class TransformerEncoderLayer(nn.Module):

    
    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        
        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)  # 1024 * 4 = 4096
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
  
        # Attention block with residual
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        
        # MLP block with residual
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        
        return x


class VisionTransformer(nn.Module):

    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 26,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2  # 196
        
        # ═══════════════════════════════════════════════════════════════
        # PATCH EMBEDDING
        # ═══════════════════════════════════════════════════════════════
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        
        # ═══════════════════════════════════════════════════════════════
        # CLASS TOKEN ([CLS])
        # ═══════════════════════════════════════════════════════════════
        # Learnable embedding prepended to patch sequence
        # Initialized as zeros, learned during training
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # ═══════════════════════════════════════════════════════════════
        # POSITIONAL EMBEDDINGS
        # ═══════════════════════════════════════════════════════════════
        # Learnable position embeddings for 197 tokens (196 patches + 1 CLS)
        # Initialized with truncated normal distribution
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Dropout after adding positional embeddings
        self.pos_drop = nn.Dropout(dropout)
        
        # ═══════════════════════════════════════════════════════════════
        # TRANSFORMER ENCODER
        # ═══════════════════════════════════════════════════════════════
        # Stochastic depth: linearly increase drop_path from 0 to drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # ═══════════════════════════════════════════════════════════════
        # CLASSIFICATION HEAD
        # ═══════════════════════════════════════════════════════════════
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using truncated normal distribution."""
        # Positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Class token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Linear layers and LayerNorm
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
        # Special initialization for output projection in attention
        # and second layer in MLP (improves training stability)
        for layer in self.encoder_layers:
            nn.init.trunc_normal_(layer.attn.proj.weight, std=0.02)
            nn.init.trunc_normal_(layer.mlp.fc2.weight, std=0.02)
            
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head.
        
        Args:
            x: [B, C, H, W] = [B, 3, 224, 224]
            
        Returns:
            [B, embed_dim] = [B, 1024] (CLS token representation)
        """
        B = x.shape[0]
        
        # Patch embedding: [B, 3, 224, 224] → [B, 196, 1024]
        x = self.patch_embed(x)
        
        # Expand CLS token for batch: [1, 1, 1024] → [B, 1, 1024]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Prepend CLS token: [B, 196, 1024] → [B, 197, 1024]
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
            
        # Final layer norm
        x = self.norm(x)
        
        # Return only CLS token representation
        return x[:, 0]  # [B, 1024]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass.
        
        Args:
            x: [B, C, H, W] = [B, 3, 224, 224]
            
        Returns:
            [B, num_classes] = [B, 26] (logits, not probabilities)
        """
        x = self.forward_features(x)  # [B, 1024]
        x = self.head(x)  # [B, 26]
        return x
    
    def get_attention_maps(self, x: torch.Tensor) -> list:
        """
        Get attention maps from all layers (for visualization).
        
        Returns:
            List of attention maps, each [B, num_heads, N, N]
        """
        attention_maps = []
        
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for layer in self.encoder_layers:
            # Get attention weights before softmax
            attn_layer = layer.attn
            qkv = attn_layer.qkv(layer.norm1(x))
            qkv = qkv.reshape(B, x.shape[1], 3, attn_layer.num_heads, attn_layer.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * attn_layer.scale
            attn = attn.softmax(dim=-1)
            attention_maps.append(attn.detach())
            
            # Continue forward pass
            x = layer(x)
            
        return attention_maps


# ═══════════════════════════════════════════════════════════════════════════
# MODEL FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def vit_large_patch16_224(num_classes: int = 26, **kwargs) -> VisionTransformer:
    """
    ViT-Large/16 model for 224x224 images.
    
    Architecture:
    - Patch size: 16x16
    - Hidden dim: 1024
    - Layers: 24
    - Heads: 16
    - MLP dim: 4096
    - Parameters: ~304M
    """
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs
    )


def vit_base_patch16_224(num_classes: int = 26, **kwargs) -> VisionTransformer:
    """
    ViT-Base/16 model (smaller, for comparison).
    
    Parameters: ~86M
    """
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        **kwargs
    )


def vit_huge_patch14_224(num_classes: int = 26, **kwargs) -> VisionTransformer:
    """
    ViT-Huge/14 model (largest, for B200).
    
    Parameters: ~632M
    """
    return VisionTransformer(
        image_size=224,
        patch_size=14,
        num_classes=num_classes,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs
    )


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: VisionTransformer):
    """Print detailed model summary."""
    print("=" * 60)
    print("VISION TRANSFORMER MODEL SUMMARY")
    print("=" * 60)
    
    print(f"\nArchitecture:")
    print(f"  Image size:    {model.patch_embed.image_size}")
    print(f"  Patch size:    {model.patch_embed.patch_size}")
    print(f"  Num patches:   {model.num_patches}")
    print(f"  Embed dim:     {model.embed_dim}")
    print(f"  Num layers:    {len(model.encoder_layers)}")
    print(f"  Num heads:     {model.encoder_layers[0].attn.num_heads}")
    print(f"  MLP dim:       {model.encoder_layers[0].mlp.fc1.out_features}")
    print(f"  Num classes:   {model.num_classes}")
    
    total_params = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Patch embed:   {count_parameters(model.patch_embed):,}")
    print(f"  CLS token:     {model.cls_token.numel():,}")
    print(f"  Pos embed:     {model.pos_embed.numel():,}")
    print(f"  Encoder:       {sum(count_parameters(l) for l in model.encoder_layers):,}")
    print(f"  Norm:          {count_parameters(model.norm):,}")
    print(f"  Head:          {count_parameters(model.head):,}")
    print(f"  ─────────────────────────")
    print(f"  TOTAL:         {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Estimate memory
    param_bytes = total_params * 4  # FP32
    print(f"\nMemory (FP32):   {param_bytes / 1e9:.2f} GB")
    print(f"Memory (BF16):   {param_bytes / 2 / 1e9:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    # Test the model
    print("Creating ViT-Large/16 model...")
    model = vit_large_patch16_224(num_classes=26, drop_path_rate=0.2)
    model_summary(model)
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output (logits) sample: {y[0, :5]}")
    
    # Test attention maps
    print("\nTesting attention map extraction...")
    attn_maps = model.get_attention_maps(x)
    print(f"Number of layers: {len(attn_maps)}")
    print(f"Attention map shape: {attn_maps[0].shape}")  # [B, heads, N, N]
