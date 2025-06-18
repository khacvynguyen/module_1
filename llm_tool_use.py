import arxiv
import traceback
import json
import os
from openai import OpenAI
from typing import List
from dotenv import load_dotenv

load_dotenv()


PAPER_DIR = "data/arxiv"


def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        List of paper IDs found in the search
    """
    
    # Use arxiv to find the papers 
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)
    
    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info  
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info
    
    # Save updated papers_info to json file
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)
    
    print(f"Results are saved in: {file_path}")
    
    return paper_ids


def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.
    
    Args:
        paper_id: The ID of the paper to look for
        
    Returns:
        JSON string with paper information if found, error message if not found
    """
 
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
    
    return f"There's no saved information related to paper {paper_id}."


# Here are the schema of each tool which you will provide to the LLM.
ARXIV_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search for papers on arXiv based on a topic and store their information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to search for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to retrieve",
                        "default": 5
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_info",
            "description": "Search for information about a specific paper across all topic directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "The ID of the paper to look for"
                    }
                },
                "required": ["paper_id"]
            }
        }
    }
]





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


# The chatbot handles the user's queries one by one, but it does not persist memory across the queries.
# Query Processing

client = OpenAI(
    api_key = os.getenv("GOOGLE_API_KEY"),
    base_url = os.getenv("GOOGLE_OPENAI_API_ENDPOINT")
)

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
