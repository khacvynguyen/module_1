import traceback
import json
import os
import logging
import asyncio
from typing import List, Callable, Optional, Dict, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv

from tools.arxiv_research import ARXIV_TOOL_SCHEMA, execute_tool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_MAX_TOKENS = 2024
MAX_TOOL_CALL_ITERATIONS = 10  # Prevent infinite loops

load_dotenv()

DEFAULT_LLM_CLIENT = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_API_URL")
)


async def process_query(
    query: str,
    client: AsyncOpenAI = DEFAULT_LLM_CLIENT,
    model: str = DEFAULT_MODEL,
    tools: List[Dict[str, Any]] = ARXIV_TOOL_SCHEMA,
    tool_executor: Callable = execute_tool,
    max_iterations: int = MAX_TOOL_CALL_ITERATIONS
) -> Optional[str]:
    """
    Process a query using LLM with tool calling capabilities.
    
    Args:
        query: The user's query string
        client: AsyncOpenAI client instance
        model: Model name to use
        tools: List of tool schemas
        tool_executor: Function to execute tools
        max_iterations: Maximum number of tool call iterations
        
    Returns:
        Final response content or None if processing failed
    """
    if not query.strip():
        logger.warning("Empty query provided")
        return None
        
    messages = [{'role': 'user', 'content': query}]
    iteration_count = 0
    
    try:
        while iteration_count < max_iterations:
            iteration_count += 1
            logger.debug(f"Starting iteration {iteration_count}")
            
            # Get LLM response
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=DEFAULT_MAX_TOKENS
            )
            
            if not response or not response.choices:
                logger.error("No response from LLM")
                return None
            
            response_message = response.choices[0].message
            messages.append(response_message)
            
            # Handle tool calls
            if response_message.tool_calls:
                logger.info(f"Processing {len(response_message.tool_calls)} tool calls")
                
                for tool_call in response_message.tool_calls:
                    tool_id = tool_call.id
                    tool_name = tool_call.function.name
                    
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse tool arguments: {e}")
                        continue
                    
                    logger.info(f"Calling tool {tool_name} with args {tool_args}")
                    
                    try:
                        # If tool_executor is async, await it; otherwise call it normally
                        if asyncio.iscoroutinefunction(tool_executor):
                            result = await tool_executor(tool_name, tool_args)
                        else:
                            result = tool_executor(tool_name, tool_args)
                        
                        if result is None:
                            result = "The operation completed but didn't return any results."
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {str(e)}")
                        result = f"Error executing tool {tool_name}: {str(e)}"

                    messages.append({
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": str(result)
                    })
                
                # Continue with next iteration to get LLM response
                continue
            
            # No tool calls, return final response
            final_content = response_message.content
            if final_content:
                logger.info("Query processing completed successfully")
                return final_content
            else:
                logger.warning("Empty response content received")
                return None
                
    except Exception as e:
        logger.error(f"Error during query processing: {str(e)}")
        logger.debug(traceback.format_exc())
        return None
    
    logger.warning(f"Maximum iterations ({max_iterations}) reached")
    return None


async def chat_loop(
    query_executor: Callable = process_query,
) -> None:
    """Interactive chat loop for processing queries."""
    print("Type your queries or 'quit' to exit.")
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                print("Please enter a query.")
                continue
            
            result = await query_executor(query)
            if result:
                print(f"\nResponse: {result}")
            else:
                print("\nNo response generated. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Unexpected error in chat loop: {str(e)}")
            print(f"\nAn error occurred: {str(e)}")


if __name__ == "__main__":
    import fire
    fire.Fire()
