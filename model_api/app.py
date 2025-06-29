from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
from typing import Dict, Any, Optional, List
import os

app = FastAPI(title="GPT-Neo-125M API", version="1.0.0")

# Global variables for model and tokenizer
model = None
tokenizer = None

# === Internal request/response model ===
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    do_sample: bool = True

class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    model_name: str

# === Load model ===
def load_model():
    global model, tokenizer
    model_path = "./gpt-neo-125m"

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Model loaded successfully!")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    load_model()

# === Health and info endpoints ===
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": str(model.device) if model else None
    }

@app.get("/model-info")
async def get_model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": "gpt-neo-125m",
        "model_type": type(model).__name__,
        "device": str(model.device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

@app.get("/")
async def root():
    return {
        "message": "GPT-Neo-125M API",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "generate": "/generate",
            "completions": "/v1/completions",
            "docs": "/docs"
        }
    }

# === Main generation logic ===
@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        inputs = tokenizer.encode(request.prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip prompt from result
        if generated_text.startswith(request.prompt):
            generated_text = generated_text[len(request.prompt):].strip()

        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            model_name="gpt-neo-125m"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# === OpenAI-compatible /v1/completions ===
@app.post("/v1/completions")
async def openai_compatible_completion(request: Request):
    data = await request.json()

    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 1.0)
    top_p = data.get("top_p", 0.9)

    generate_request = GenerateRequest(
        prompt=prompt,
        max_length=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True
    )

    response = await generate_text(generate_request)

    return {
        "id": "cmpl-local-001",
        "object": "text_completion",
        "choices": [{
            "text": response.generated_text,
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

# === Run server ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
