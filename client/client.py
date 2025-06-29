import requests
import json
import time
import os
from langfuse import observe

# Configure Langfuse environment variables - these should match docker-compose.yml
os.environ['LANGFUSE_PUBLIC_KEY'] = 'pk-lf-cf2554fb-652d-4b1a-b126-6236800a9408'
os.environ['LANGFUSE_SECRET_KEY'] = 'sk-lf-7d328fde-7986-4db9-8a52-0ec3e544f179'
os.environ['LANGFUSE_HOST'] = 'http://langfuse-web:3000'

def wait_for_service(url, max_retries=30, delay=2):
    """Wait for a service to be ready"""
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Service at {url} is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        print(f"⏳ Waiting for service at {url}... (attempt {i+1}/{max_retries})")
        time.sleep(delay)
    return False

@observe()
def make_inference_request(prompt, max_tokens=50, temperature=0.8):
    """Make a direct HTTP request to the model API"""
    url = "http://model-api:8000/v1/completions"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["text"]
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

@observe()
def story():
    """Generate a story using the local model"""
    prompt = "Tell me a story about a robot learning to paint"
    return make_inference_request(prompt, max_tokens=80, temperature=0.8)

@observe()
def programming_advice():
    """Get programming advice using the local model"""
    prompt = "What is the best programming language for beginners? Explain briefly."
    return make_inference_request(prompt, max_tokens=60, temperature=0.5)

@observe()
def main():
    # Wait for services to be ready
    print("🚀 Starting LLM client with monitoring...")
    
    # Wait for model API
    if not wait_for_service("http://model-api:8000/health"):
        print("❌ Model API service is not ready!")
        return
    
    # Wait for Langfuse web
    if not wait_for_service("http://langfuse-web:3000"):
        print("❌ Langfuse web service is not ready!")
        return
    
    print("🎯 All services are ready! Starting inference...")
    
    try:
        # Test multiple inference requests
        print("\n📝 Test 1: Story generation")
        result1 = story()
        if result1:
            print(f"✅ Generated story: {result1[:100]}...")
        
        print("\n📝 Test 2: Programming advice")
        result2 = programming_advice()
        if result2:
            print(f"✅ Generated advice: {result2[:100]}...")
        
        print("\n📝 Test 3: Direct request")
        result3 = make_inference_request("Explain quantum computing in simple terms", max_tokens=50, temperature=0.7)
        if result3:
            print(f"✅ Generated explanation: {result3[:100]}...")
        
        print("\n🎉 All inference tests completed!")
        print("📊 Check Langfuse dashboard at http://localhost:3000 to see the traces!")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")

if __name__ == "__main__":
    main()
