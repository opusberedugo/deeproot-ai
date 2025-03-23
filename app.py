from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import Transformer, ModelArgs
import torch
import os
from typing import List

app = FastAPI(title="AI Text Generation API", description="Simple API for text generation")

# Load config
config_path = os.path.join("configs", "config.json")
args = ModelArgs.from_json(config_path)

# Initialize model
print("Loading model...")
model = Transformer(args)

try:
    # Load model weights (replace with your actual model file)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load model")

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50
    temperature: float = 0.7
    top_k: int = 50

@app.post("/generate", summary="Generate text from a prompt")
async def generate_text(request: GenerationRequest):
    """
    Generate text based on the provided prompt.

    - **prompt**: Input text to generate from
    - **max_length**: Maximum number of tokens to generate (default: 50)
    - **temperature**: Sampling temperature (default: 0.7)
    - **top_k**: Top-k sampling (default: 50)
    """
    try:
        # Convert prompt to input IDs (simplified for demo)
        input_ids = torch.tensor([[0]])  # Replace with actual tokenization
        
        # Generate text
        with torch.no_grad():
            output = model(input_ids)
        
        # Decode output (simplified for demo)
        generated_text = request.prompt[:10] + "..."  # Replace with actual decoding
        
        return {"result": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", summary="Health check endpoint")
async def health_check():
    """
    Check if the API is running.
    """
    return {"status": "ok", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)