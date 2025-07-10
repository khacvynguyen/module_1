import os
import tiktoken
from structlog import get_logger
from typing import Generator, AsyncGenerator
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai

load_dotenv()

# Configure Gemini if API key is available
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

logger = get_logger(__name__)
DEFAULT_ENCODE_MODEL = tiktoken.encoding_for_model("gpt-4o-mini")

LLM_TIMEOUT = os.getenv("LLM_TIMEOUT", 60)
LLM_MAX_RETRIES = os.getenv("LLM_MAX_RETRIES", 3)

LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

DEFAULT_CLIENT = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL
)

DEFAULT_ASYNC_CLIENT = AsyncOpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL
)


if not DEFAULT_CLIENT:
    logger.warning("Default LLM client is not initialized")

if not DEFAULT_ASYNC_CLIENT:
    logger.warning("Default LLM async client is not initialized")


def get_length(text, encode_model="gpt-4o-mini"):
    if encode_model != "gpt-4o-mini":
        enc = tiktoken.encoding_for_model(encode_model)
    else:
        enc = DEFAULT_ENCODE_MODEL
    return len(enc.encode(text))


def wrap_stream(stream) -> Generator:
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

async def async_wrap_stream(stream) -> AsyncGenerator:
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


def gemini_completion(
    model: str,
    messages: list[dict],
    **kwargs
) -> dict:
    """
    Gemini completion function that handles Google Gemini models and File objects.
    """
    try:
        # Extract temperature and max_tokens from kwargs
        generation_config = {}
        if 'temperature' in kwargs:
            generation_config['temperature'] = kwargs['temperature']
        if 'max_tokens' in kwargs:
            generation_config['max_output_tokens'] = kwargs['max_tokens']
        
        # Create the Gemini model
        gemini_model = genai.GenerativeModel(model_name=model)
        
        # Convert messages to Gemini format
        # For now, we'll handle the simple case where there's one user message
        # with content that may include File objects
        if len(messages) == 1 and messages[0]["role"] == "user":
            content = messages[0]["content"]
            if isinstance(content, list):
                # Handle mixed content (File objects and text)
                response = gemini_model.generate_content(
                    content,
                    generation_config=generation_config
                )
            else:
                # Handle simple text content
                response = gemini_model.generate_content(
                    content,
                    generation_config=generation_config
                )
        else:
            # For more complex conversation histories, we'd need to convert properly
            # For now, just handle the simple case
            raise NotImplementedError("Multi-turn conversations not yet supported for Gemini")
        
        return {"content": response.text}
    except Exception as e:
        logger.error(f"Error in Gemini completion: {e}", exc_info=True)
        return {"content": f"Error in Gemini completion: {e}"}


async def gemini_async_completion(
    model: str,
    messages: list[dict],
    **kwargs
) -> dict:
    """
    Async Gemini completion function that handles Google Gemini models and File objects.
    """
    try:
        # Extract temperature and max_tokens from kwargs
        generation_config = {}
        if 'temperature' in kwargs:
            generation_config['temperature'] = kwargs['temperature']
        if 'max_tokens' in kwargs:
            generation_config['max_output_tokens'] = kwargs['max_tokens']
        
        # Create the Gemini model
        gemini_model = genai.GenerativeModel(model_name=model)
        
        # Convert messages to Gemini format
        # For now, we'll handle the simple case where there's one user message
        # with content that may include File objects
        if len(messages) == 1 and messages[0]["role"] == "user":
            content = messages[0]["content"]
            if isinstance(content, list):
                # Handle mixed content (File objects and text)
                response = await gemini_model.generate_content_async(
                    content,
                    generation_config=generation_config
                )
            else:
                # Handle simple text content
                response = await gemini_model.generate_content_async(
                    content,
                    generation_config=generation_config
                )
        else:
            # For more complex conversation histories, we'd need to convert properly
            # For now, just handle the simple case
            raise NotImplementedError("Multi-turn conversations not yet supported for Gemini")
        
        return {"content": response.text}
    except Exception as e:
        logger.error(f"Error in async Gemini completion: {e}", exc_info=True)
        return {"content": f"Error in async Gemini completion: {e}"}


def llm_completion(
    model: str,
    messages: list[dict],
    client: OpenAI = DEFAULT_CLIENT,
    **kwargs
) -> str:
    """
    LLM completion with backoff and stream support.
    Routes to appropriate handler based on model name.
    """
    # Route to Gemini handler if it's a Gemini model
    if "gemini" in model.lower():
        return gemini_completion(model, messages, **kwargs)
    
    # Original OpenAI handling
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        if kwargs.get("stream"):
            return {"content": wrap_stream(response)}
        else:
            return {"content": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"Error in LLM completion: {e}", exc_info=True)
        return {"content": f"Error in LLM completion: {e}"}


async def llm_async_completion(
    model: str,
    messages: list[dict],
    client: AsyncOpenAI = DEFAULT_ASYNC_CLIENT,
    **kwargs
) -> str:
    """
    LLM async completion
    Routes to appropriate handler based on model name.
    """
    # Route to Gemini handler if it's a Gemini model
    if "gemini" in model.lower():
        return await gemini_async_completion(model, messages, **kwargs)
    
    # Original OpenAI handling
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        if kwargs.get("stream"):    
            return {"content": async_wrap_stream(response)}
        else:
            return {"content": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"Error in LLM async completion: {e}", exc_info=True)
        return {"content": f"Error in LLM async completion: {e}"} 


def llm_model_list(client: OpenAI = DEFAULT_CLIENT):
    return [model.id for model in client.models.list()]


@retry(wait=wait_exponential(multiplier=2, min=1, max=LLM_TIMEOUT), stop=stop_after_attempt(LLM_MAX_RETRIES))
def llm_completion_with_backoff(**kwargs):
    return llm_completion(**kwargs)


@retry(wait=wait_exponential(multiplier=2, min=1, max=LLM_TIMEOUT), stop=stop_after_attempt(LLM_MAX_RETRIES))
async def llm_async_completion_with_backoff(**kwargs):
    return await llm_async_completion(**kwargs)


def llm_runall(**kwargs):
    try:
        return llm_completion_with_backoff(**kwargs)
    except Exception as e:
        print(f"ERROR || {e}")
        return {"content": f"ERROR || {e}"}
    

async def llm_async_runall(**kwargs):
    try:
        return await llm_async_completion_with_backoff(**kwargs)
    except Exception as e:
        logger.error(f"ERROR || {e}")
        return {"content": f"ERROR || {e}"}


def test_api_call(model: str = "gpt-4o", stream: bool = False):
    messages = [{"role": "user", "content": "hello"}]
    response = llm_completion(messages=messages, model=model, stream=stream)
    if stream:
        for chunk in response["content"]:
            print(chunk)
    else:
        print(response["content"])


async def test_async_api_call(model: str = "gpt-4o", stream: bool = False):
    messages = [{"role": "user", "content": "hello"}]
    response = await llm_async_completion(messages=messages, model=model, stream=stream)
    if stream:
        async for chunk in response["content"]:
            print(chunk)
    else:
        print(response["content"])


def test_custom_client(model: str = "gemini-2.5-flash", **kwargs):
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL")
    )
    messages = [{"role": "user", "content": "what is capital of france?"}]
    response = llm_completion(messages=messages, model=model, client=client, **kwargs)
    print(response["content"])


def test_model_list():
    return llm_model_list()


if __name__ == "__main__":
    import fire
    fire.Fire()