import traceback
import json
import os
from openai import OpenAI
from typing import List
from tools.arxiv_research import (
    ARXIV_TOOL_SCHEMA,
    search_papers,
    extract_info
)
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key = os.getenv("GOOGLE_API_KEY"),
    base_url = os.getenv("GOOGLE_OPENAI_API_ENDPOINT")
)

MAPPING_TOOL_FUNCTION = {
    "search_papers": search_papers,
    "extract_info": extract_info
}


def execute_tool(tool_name, tool_args):
    
    result = MAPPING_TOOL_FUNCTION[tool_name](**tool_args)

    if result is None:
        result = "The operation completed but didn't return any results."
        
    elif isinstance(result, list):
        result = ', '.join(result)
        
    elif isinstance(result, dict):
        # Convert dictionaries to formatted JSON strings
        result = json.dumps(result, indent=2)
    
    else:
        # For any other type, convert using str()
        result = str(result)
    return result


def process_query(
    query: str,
    model: str = "gemini-2.5-flash"
):
    messages = [{'role': 'user', 'content': query}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=ARXIV_TOOL_SCHEMA,
        tool_choice="auto",
        max_tokens = 2024
    )
    
    if not response:
        print("No response from LLM")
        return
    
    # Process the model's response
    response_message = response.choices[0].message
    messages.append(response_message)
    process_query = True

    # import pdb; pdb.set_trace()

    while process_query:
        # Handle tool calls
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                tool_id = tool_call.id
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                print(f"Calling tool {tool_name} with args {tool_args}")
                
                try:
                    result = execute_tool(tool_name, tool_args)
                except Exception as e:
                    print(f"Error executing tool {tool_name}: {str(e)}")
                    if tool_name == "search_papers":
                        print("Using sample data for search_papers")
                        result = str(json.loads(open("assets/sample_info.json", "r").read()))
                    else:
                        result = "The operation completed but didn't return any results."

                messages.append({
                    "tool_call_id": tool_id,
                    "role": "tool",
                    "name": tool_name,
                    "content": result
                })

            # continue call LLM with the new messages
            response = client.chat.completions.create(
                model = model, 
                tools = ARXIV_TOOL_SCHEMA,
                messages = messages
            )
            if not response:
                print("No response from LLM")
                break
    
            response_message = response.choices[0].message
            messages.append(response_message)

            if not response_message.tool_calls:
                print(response_message.content)
                process_query = False

        else:
            print(response_message.content)
            process_query = False
                
            if len(response.content) == 1:
                process_query = False


#  Chat Loop
def chat_loop():
    print("Type your queries or 'quit' to exit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
    
            process_query(query)
            print("\n")
        except Exception as e:
            print(f"\nError: {str(e)}")
            traceback.print_exc()


if __name__ == "__main__":
    import fire
    fire.Fire()
