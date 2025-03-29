from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from model import AgriculturalTransformer, ModelArgs
from transformers import AutoTokenizer
import os

app = FastAPI(title="AgriAI Assistant", description="Agricultural Recommendation System")

# Load configuration and model
try:
    # Ensure config directory exists
    if not os.path.exists(os.path.join("configs")):
        os.makedirs(os.path.join("configs"))
        
    config_path = os.path.join("configs", "config.json")
    model_path = "model.pth"
    
    # If the model file doesn't exist, look in the models directory
    if not os.path.exists(model_path):
        model_path = os.path.join("models", "agricultural_transformer.pth")
    
    # Load model arguments
    args = ModelArgs.from_json(config_path)
    
    # Create tokenizer first to get vocab size
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure vocabulary size matches tokenizer
    if args.vocab_size is None or args.vocab_size != len(tokenizer):
        args.vocab_size = len(tokenizer)
        print(f"Updated vocabulary size to match tokenizer: {args.vocab_size}")
    
    # Create and load model
    model = AgriculturalTransformer(args)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Use the model's tokenizer for consistency
    tokenizer = model.tokenizer
    
    print(f"Model loaded successfully from {model_path}")
    print(f"Model vocabulary size: {model.args.vocab_size}")
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 128
    temperature: float = 0.7
    top_k: int = 50

def generate_response(prompt: str, max_length: int = 128, temperature: float = 0.7, top_k: int = 50):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True, padding=True)
    input_ids = inputs.input_ids
    
    # For simple greedy generation
    if temperature == 0:
        with torch.no_grad():
            # Initial forward pass
            outputs = model(input_ids)
            # Generate sequence
            generated = input_ids
            
            for _ in range(max_length):
                # Get model predictions
                outputs = model(generated)
                # Get next token prediction (greedy)
                next_token_logits = outputs[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                # Add to sequence
                generated = torch.cat([generated, next_token], dim=-1)
                # Stop if end token
                if next_token.item() == tokenizer.eos_token_id:
                    break
    # For sampling with temperature
    else:
        with torch.no_grad():
            # Initial forward pass
            outputs = model(input_ids)
            # Generate sequence
            generated = input_ids
            
            for _ in range(max_length):
                # Get model predictions
                outputs = model(generated)
                # Get next token prediction with temperature
                next_token_logits = outputs[:, -1, :] / max(0.1, temperature)
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[next_token_logits < indices_to_remove] = float('-inf')
                
                # Convert to probabilities
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                
                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Add to sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Stop if end token
                if next_token.item() == tokenizer.eos_token_id:
                    break
    
    # Decode generated sequence
    return tokenizer.decode(generated[0], skip_special_tokens=True)

@app.post("/recommend", summary="Get agricultural recommendations")
async def get_recommendation(request: GenerationRequest):
    try:
        recommendation = generate_response(
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k
        )
        return {"recommendation": recommendation}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error generating recommendation: {error_details}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "active", "model": "AgriAI 1.0", "vocab_size": model.args.vocab_size}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)