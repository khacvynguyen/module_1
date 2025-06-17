import os
import time
from typing import Dict, get_type_hints

import tiktoken
import anthropic
import google.generativeai as google_genai
from dotenv import load_dotenv
from openai import OpenAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from tenacity import retry, stop_after_attempt, wait_exponential

enc = tiktoken.encoding_for_model("gpt-4o-mini")


load_dotenv()

# Set your API keys here
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TOGETHERAI_API_KEY = os.getenv("TOGETHERAI_API_KEY", "")


# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# Initialize TogetherAI client
together_client = OpenAI(
    api_key=TOGETHERAI_API_KEY,
    base_url="https://api.together.xyz/v1",
)

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Configure Google Generative AI
google_genai.configure(api_key=GOOGLE_API_KEY)

DEFAULT_SAFETY_SETTINGS: Dict[HarmCategory, HarmBlockThreshold] = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
}
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.7
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 60))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 3))


def format_google_messages(messages):
    """Convert OpenAI-style messages to Google's format"""
    formatted_messages = []
    system_message = None

    for msg in messages:
        if msg["role"] == "system":
            system_message = msg["content"]
        else:
            formatted_messages.append({"role": msg["role"], "parts": [msg["content"]]})

    return system_message, formatted_messages


def gemini_stream_wrapper(stream):
    for chunk in stream:
        yield chunk.text 


def ai_completion(**kwargs):
    """
    # Example usage
    params = {
        "model": "gemini-pro",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a story about a magic backpack."}
        ],
        "max_tokens": 1000,
        "temperature": 0.7,
    }

    params = {
        "api_key: "sk...",
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a story about a magic backpack."}
        ],
        "max_tokens": 1000,
        "temperature": 0.7,
        "stream": True
    }
    """
    try:
        # signal.alarm(60)

        model_name = kwargs.get("model", "")
        if not model_name:
            raise ValueError("Model name is required")
        
        provider = get_provider_client(model_name)

        if provider == "openai":
            # Initialize OpenAI client
            api_key = kwargs.pop("api_key", OPENAI_API_KEY)
            openai_client = OpenAI(api_key=api_key)
            response = openai_client.chat.completions.create(**kwargs)
            if kwargs.get("stream"):
                return {"content": response}
            
            else:
                result = {"content": response.choices[0].message.content}

        elif provider == "google":

            generation_config = dict(
                max_output_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
                temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
                response_mime_type="text/plain",
            )

            model = google_genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
            )

            # Format messages for Google's API
            system_message, chat_history = format_google_messages(kwargs["messages"])

            # Create chat session with system message if provided
            if system_message:
                model = google_genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    system_instruction=system_message,
                )

            # Start chat session with history
            chat = model.start_chat(history=chat_history[:-1] if len(chat_history) > 1 else [])

            # Send the last message
            response = chat.send_message(
                chat_history[-1]["parts"][0],
                stream=kwargs.get("stream", False)
                # safety_settings=DEFAULT_SAFETY_SETTINGS,
            )
            if kwargs.get("stream"):
                result = {"content": gemini_stream_wrapper(response)}
                
            else:
                result = {"content": response.text}

        elif provider == "anthropic":
            response = anthropic_client.messages.create(
                model=model_name,
                max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
                temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
                messages=kwargs["messages"],
            )
            result = {"content": response.content[0].text}

        else:
            raise ValueError(f"Unsupported AI provider {provider}")

        return result

    except Exception as e:
        if "Resource has been exhausted" in str(e):
            print("Google API - sleep 60 seconds")
            time.sleep(60)
            raise ValueError("Google API - Resource has been exhausted - already sleep 60 seconds")
        elif "block_reason" in str(e):
            print("Google API - blocked - change to gpt-4o-mini")
            kwargs["model"] = "gpt-4o-mini"
            return ai_completion(**kwargs)
        else:
            print(f"ERROR ai_completion - {e}")
            raise ValueError(e)


@retry(wait=wait_exponential(multiplier=2, min=1, max=LLM_TIMEOUT), stop=stop_after_attempt(LLM_MAX_RETRIES))
def ai_completion_with_backoff(**kwargs):
    return ai_completion(**kwargs)


def get_provider_client(model_name):
    if "gpt" in model_name:
        provider = "openai"
    elif "gemini" in model_name:
        provider = "google"
    elif "claude" in model_name:
        provider = "anthropic"
    else:
        provider = model_name
    return provider


def ai_runall(**kwargs):
    try:
        return ai_completion_with_backoff(**kwargs)
    except Exception as e:
        print(f"ERROR || {e}")
        return {"content": f"ERROR || {e}"}


def ai_model_list():
    return {
        "openai": [model.id for model in openai_client.models.list()],
        "google": [model.name.split("/")[-1] for model in google_genai.list_models()],
        "anthropic": list(get_type_hints(anthropic_client.messages.create)["model"].__args__[1].__args__),
    }


def truncate_text(text, max_tokens=8000):
    tokens = enc.encode(text)
    return enc.decode(tokens[:max_tokens])


def get_length(text):
    return len(enc.encode(text))


def test_api_call():
    messages = [{"role": "user", "content": "hello"}]
    ai_completion(messages=messages, model="gpt-4o")


if __name__ == "__main__":
    # ## Change your role here:
    # role = "Sales representative"
    # # reply "exit" to exit

    # start_conversation(role=role)
    # test_api_call()
    
    import fire
    fire.Fire()
