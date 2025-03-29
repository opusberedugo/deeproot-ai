import torch
import torch.nn as nn
from dataclasses import dataclass
import json
import os
import math
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

@dataclass
class ModelArgs:
    dim: int = 768
    n_layers: int = 6
    n_heads: int = 12
    vocab_size: int = None  # Will be set based on the tokenizer
    max_seq_len: int = 256
    inter_dim: int = 3072
    dropout: float = 0.1

    @classmethod
    def from_json(cls, config_path: str):
        try:
            with open(config_path) as f:
                config = json.load(f)
            return cls(**config)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading config: {e}")
            print("Using default configuration.")
            return cls()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Calculate positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # Add batch dimension [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # Add positional encoding to input embeddings
        return x + self.pe[:, :x.size(1), :]

class AgriculturalTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # Set up tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Update args.vocab_size based on the tokenizer
        if self.args.vocab_size is None or self.args.vocab_size != len(self.tokenizer):
            self.args.vocab_size = len(self.tokenizer)
        
        # Model components with correct vocabulary size
        self.embedding = nn.Embedding(self.args.vocab_size, args.dim)
        self.pos_encoder = PositionalEncoding(args.dim, args.max_seq_len)
        
        # Simple encoder-only transformer (more suitable for this task)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.dim,
            nhead=args.n_heads,
            dim_feedforward=args.inter_dim,
            dropout=args.dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=args.n_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(args.dim, self.args.vocab_size)
    
    def forward(self, x):
        # Create source mask for padding tokens
        src_key_padding_mask = (x == self.tokenizer.pad_token_id)
        
        # Create attention mask to prevent looking ahead
        seq_len = x.size(1)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        attn_mask = attn_mask.to(x.device)
        
        # Embed tokens
        x = self.embedding(x) * math.sqrt(self.args.dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(
            x, 
            mask=attn_mask, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Project to vocabulary
        return self.output_projection(output)

class AgricultureDataset(Dataset):
    def __init__(self, tokenizer, max_length=128):
        self.data = [
            ("Soil type: sandy, Rainfall: low", "Recommended crops: millet, sorghum, and cassava."),
            ("Region: tropical, PH: acidic", "Best crops: coffee, tea, and bananas."),
            ("Climate: arid, Water: scarce", "Suggested crops: cactus, agave, and date palms.")
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, response = self.data[idx]
        
        # For simplicity, we'll use an encoder-only approach
        # We concatenate the prompt and response for training
        full_text = f"{prompt} {response}"
        
        # Tokenize the text
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get input IDs and create target IDs (shifted by 1)
        input_ids = encodings.input_ids.squeeze(0)
        target_ids = input_ids.clone()
        
        # For language modeling, targets are the input shifted by 1
        target_ids[:-1] = input_ids[1:]
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids
        }

def train_model():
    # Create a tokenizer first to get its vocab size
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create config directory if needed
    config_path = "configs/config.json"
    if not os.path.exists(os.path.dirname(config_path)):
        os.makedirs(os.path.dirname(config_path))
        
        # Create default config with correct vocab size
        default_config = {
            "dim": 768,
            "n_layers": 6,
            "n_heads": 12,
            "vocab_size": len(tokenizer),  # Use the actual tokenizer vocab size
            "max_seq_len": 256, 
            "inter_dim": 3072,
            "dropout": 0.1
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
    
    # Load model arguments
    args = ModelArgs.from_json(config_path)
    
    # Ensure vocab_size is set correctly
    if args.vocab_size is None or args.vocab_size != len(tokenizer):
        args.vocab_size = len(tokenizer)
    
    # Create model
    model = AgriculturalTransformer(args)
    tokenizer = model.tokenizer
    
    # Create dataset and dataloader
    dataset = AgricultureDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    print(f"Training on device: {device}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Vocabulary size: {args.vocab_size}")
    
    # Training loop
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in dataloader:
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids)
            
            # Calculate loss
            # Reshape outputs to [batch_size*seq_len, vocab_size]
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = target_ids.view(-1)
            loss = criterion(outputs_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/3 - Loss: {avg_loss:.4f}")
    
    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), "models/agricultural_transformer.pth")
    print("Model trained and saved successfully!")
    
    # Generate a sample output
    model.eval()
    test_prompt = "Soil type: clay, Rainfall: high"
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
    
    # Generation parameters
    max_length = 50
    generated = input_ids
    
    # Simple greedy decoding
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode and print result
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\nTest prompt: {test_prompt}")
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    train_model()