#!/usr/bin/env python3
"""
Simple example showing how to use the LocalModelClient
"""

from client import LocalModelClient

def main():
    # Initialize the client
    client = LocalModelClient()
    
    # Simple text generation
    print("🤖 Generating text with local GPT-Neo-125M model...")
    result = client.generate_text(
        prompt="Write a short story about a robot learning to paint",
        max_tokens=80,
        temperature=0.8
    )
    
    if "error" not in result:
        print(f"✅ Generated text: {result['generated_text']}")
        print(f"📊 Tokens used: {result['usage'].total_tokens}")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Test with system message
    print("\n🎭 Testing with system message...")
    result2 = client.generate_text(
        prompt="What is the best programming language for beginners?",
        max_tokens=50,
        temperature=0.3,
        system_message="You are a helpful programming tutor."
    )
    
    if "error" not in result2:
        print(f"✅ Generated text: {result2['generated_text']}")
        print(f"📊 Tokens used: {result2['usage'].total_tokens}")
    else:
        print(f"❌ Error: {result2['error']}")

if __name__ == "__main__":
    main() 