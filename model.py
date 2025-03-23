import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class ModelArgs:
    """
    Configuration class for the transformer model.
    """
    dim: int = 512  # Model dimension
    n_layers: int = 4  # Number of transformer layers
    n_heads: int = 8  # Number of attention heads
    vocab_size: int = 10000  # Vocabulary size
    max_seq_len: int = 512  # Maximum sequence length
    inter_dim: int = 2048  # Intermediate dimension for feed-forward layers
    dropout: float = 0.1  # Dropout rate

    @classmethod
    def from_json(cls, config_path: str):
        """
        Load model configuration from a JSON file.

        Args:
            config_path (str): Path to the JSON configuration file.

        Returns:
            ModelArgs: An instance of ModelArgs with loaded configuration.
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(**config)

class Transformer(nn.Module):
    """
    A simplified transformer model for CPU-based inference.
    """
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
                dim_feedforward=args.inter_dim,
                dropout=args.dropout,
                batch_first=True
            ) for _ in range(args.n_layers)
        ])
        
        # Output layer
        self.output = nn.Linear(args.dim, args.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for linear and embedding layers.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, seq_len, vocab_size).
        """
        # Embed input tokens
        x = self.embed(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Generate logits
        return self.output(x)

# Example usage
if __name__ == "__main__":
    # Load config from JSON
    args = ModelArgs.from_json("configs/config.json")
    
    # Initialize model
    model = Transformer(args)
    
    # Test forward pass
    input_ids = torch.tensor([[1, 2, 3, 4]])  # Example input
    output = model(input_ids)
    print("Output shape:", output.shape)