import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 512  # Model dimension
    n_layers: int = 4  # Number of transformer layers
    n_heads: int = 8  # Number of attention heads
    vocab_size: int = 10000  # Vocabulary size
    max_seq_len: int = 512  # Maximum sequence length

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # Embedding layer
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=args.dim,
                nhead=args.n_heads,
                dim_feedforward=4 * args.dim,
                batch_first=True
            ) for _ in range(args.n_layers)
        ])
        
        # Output layer
        self.output = nn.Linear(args.dim, args.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed input tokens
        x = self.embed(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Generate logits
        return self.output(x)