"""
Modular PyTorch Transformer Architecture with Layer-by-Layer Execution and Profiling.
Supports DistilBERT/BERT-style architectures and lightweight test models.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration parameters for the Transformer model."""
    model_name: str = "DistilBERT-base-split"
    vocab_size: int = 30522
    hidden_dim: int = 768
    num_layers: int = 6           # L: Total number of transformer encoder layers
    num_heads: int = 12
    intermediate_dim: int = 3072
    max_position_embeddings: int = 512
    dropout_prob: float = 0.1
    num_classes: int = 2          # e.g., binary sentiment or classification head

    @classmethod
    def create_tiny_test_config(cls, num_layers: int = 4, hidden_dim: int = 256) -> "TransformerConfig":
        """Lightweight configuration for rapid profiling and unit testing."""
        return cls(
            model_name="TinyTransformer-Test",
            vocab_size=1000,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=4,
            intermediate_dim=hidden_dim * 4,
            max_position_embeddings=256,
            dropout_prob=0.0,
            num_classes=2
        )


class MultiHeadSelfAttention(nn.Module):
    """Standard Multi-Head Self-Attention with pre-layer or post-layer norm."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Projections and head reshaping: (batch, num_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # Mask shape: (batch, 1, 1, seq_len)
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(context)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder block: Self-Attention + MLP with residual connections."""

    def __init__(self, hidden_dim: int, num_heads: int, intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-12)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-12)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN / Post-LN residual block
        attn_out = self.attention(x, attention_mask=attention_mask)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x


class TransformerEmbedding(nn.Module):
    """Word and Position Embeddings followed by LayerNorm."""

    def __init__(self, vocab_size: int, hidden_dim: int, max_position: int = 512, dropout: float = 0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(max_position, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        device = input_ids.device
        positions = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(positions)
        return self.dropout(self.norm(embeddings))


class ModularTransformer(nn.Module):
    """
    Modular Transformer supporting execution of arbitrary layer slices [start_idx, end_idx)
    with accurate runtime profiling.
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        self.config = config or TransformerConfig()
        
        self.embedding = TransformerEmbedding(
            vocab_size=self.config.vocab_size,
            hidden_dim=self.config.hidden_dim,
            max_position=self.config.max_position_embeddings,
            dropout=self.config.dropout_prob
        )
        
        # ModuleList allows accessing layers individually by index for split inference
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                intermediate_dim=self.config.intermediate_dim,
                dropout=self.config.dropout_prob
            )
            for _ in range(self.config.num_layers)
        ])
        
        # Classification head for downstream tasks
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.config.dropout_prob),
            nn.Linear(self.config.hidden_dim, self.config.num_classes)
        )
        
        self.eval()  # Default to evaluation mode for inference

    def forward_embeddings(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Runs embedding layer and returns (hidden_states, elapsed_seconds)."""
        t0 = time.perf_counter()
        hidden = self.embedding(input_ids)
        elapsed = time.perf_counter() - t0
        return hidden, elapsed

    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """Runs a single layer i and returns (hidden_states, elapsed_seconds)."""
        t0 = time.perf_counter()
        hidden = self.layers[layer_idx](hidden_states, attention_mask=attention_mask)
        elapsed = time.perf_counter() - t0
        return hidden, elapsed

    def forward_head(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Runs classification head on [CLS] token (index 0) and returns (logits, elapsed_seconds)."""
        t0 = time.perf_counter()
        cls_token = hidden_states[:, 0, :]
        logits = self.classifier(cls_token)
        elapsed = time.perf_counter() - t0
        return logits, elapsed

    def forward_full(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Full end-to-end forward pass with profiling of each individual stage.
        """
        profiles: Dict[str, float] = {}
        
        hidden, t_embed = self.forward_embeddings(input_ids)
        profiles["embeddings"] = t_embed
        
        for i, layer in enumerate(self.layers):
            hidden, t_layer = self.forward_layer(i, hidden, attention_mask)
            profiles[f"layer_{i}"] = t_layer
            
        logits, t_head = self.forward_head(hidden)
        profiles["head"] = t_head
        profiles["total"] = sum(profiles.values())
        
        return logits, profiles
