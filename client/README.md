# LiteLLM Client for Local GPT-Neo-125M Model

This client demonstrates how to use LiteLLM to inference a local GPT-Neo-125M model hosted via FastAPI.

## Setup

### 1. Start the FastAPI Server

First, make sure your FastAPI server is running:

```bash
cd model_api
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 2. Install Dependencies

```bash
cd client
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from client import LocalModelClient

# Initialize the client
client = LocalModelClient()

# Generate text
result = client.generate_text(
    prompt="Write a short poem about AI",
    max_tokens=50,
    temperature=0.8
)

print(result['generated_text'])
```

### Advanced Usage

```python
from client import LocalModelClient

client = LocalModelClient()

# With system message
result = client.generate_text(
    prompt="What is machine learning?",
    max_tokens=60,
    temperature=0.3,
    system_message="You are a helpful AI assistant."
)

# Batch generation
prompts = [
    "What is Python?",
    "Explain neural networks",
    "Tell me a programming joke"
]

results = client.batch_generate(
    prompts=prompts,
    max_tokens=40,
    temperature=0.7
)
```

### Async Usage

```python
import asyncio
from client import LocalModelClient

async def main():
    client = LocalModelClient()
    
    result = await client.generate_text_async(
        prompt="Explain quantum computing",
        max_tokens=70,
        temperature=0.6
    )
    
    print(result['generated_text'])

asyncio.run(main())
```

## Examples

### Run the Demo

```bash
python client.py
```

This will run a comprehensive demo showing:
- Basic text generation
- Async generation
- Batch generation
- Model information

### Run Simple Example

```bash
python example_usage.py
```

This runs a simple example with basic text generation.

## API Endpoints

The client connects to these FastAPI endpoints:

- `GET /health` - Health check
- `GET /model-info` - Model information
- `POST /v1/chat/completions` - LiteLLM-compatible chat completions

## Configuration

You can customize the client by passing parameters:

```python
client = LocalModelClient(
    base_url="http://localhost:8000",  # Your FastAPI server URL
    model_name="gpt-neo-125m"          # Model name
)
```

## Error Handling

The client includes comprehensive error handling:

- Connection testing on initialization
- Graceful error responses
- Detailed error messages

## Features

- ✅ LiteLLM compatibility
- ✅ Async support
- ✅ Batch generation
- ✅ System messages
- ✅ Error handling
- ✅ Connection testing
- ✅ Model information
- ✅ Usage statistics 