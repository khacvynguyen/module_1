import os
import json
import asyncio
import logging
import traceback
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from typing import List, Dict, TypedDict, Any, Literal
from llm_tool_use import chat_loop

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_MAX_TOKENS = 2024
MAX_TOOL_CALL_ITERATIONS = 10  # Prevent infinite loops


class FunctionDefinition(TypedDict):
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolDefinition(TypedDict):
    type: Literal["function"]
    function: FunctionDefinition


class MCP_ChatBot:

    def __init__(self, server_config_path: str = "server_config.json", **kwargs):

        # Initialize session and client objects

        self.server_config_path = server_config_path
        self.client = AsyncOpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url=os.getenv("GOOGLE_OPENAI_API_ENDPOINT")
        )
        assert self.client is not None, "LLM client is not initialized"

        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.model = kwargs.get("model", DEFAULT_MODEL)
        self.max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.max_iterations = kwargs.get("max_iterations", MAX_TOOL_CALL_ITERATIONS)

    async def connect_to_server(self, server_name: str, server_config: Dict[str, Any]) -> None:
        try:
            server_params = StdioServerParameters(**server_config)
            # Create a context manager for the stdio transport
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport

            # Create a context manager for the session
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions.append(session)

            # List available tools for this session
            response = await session.list_tools()
            tools = response.tools
            logger.info(f"\nConnected to {server_name} with tools: {', '.join([t.name for t in tools])}")

            for tool in tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append(
                    ToolDefinition(
                        type="function",
                        function=FunctionDefinition(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.inputSchema
                        )
                    )
                )

        except Exception as e:
            logger.error(f"Failed to connect to {server_name}: {e}", exc_info=True)
            print(traceback.format_exc())

    async def connect_to_servers(self):
        """Connect to all configured MCP servers."""
        try:
            server_config = json.loads(open(self.server_config_path).read())
            servers = server_config.get("mcpServers", {})

            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            logger.error(f"Error loading server configuration: {e}", exc_info=True)
            print(traceback.format_exc())
            raise

    # new process query function with multile sessions and tools
    async def mcp_process_query(self, query):

        if not query.strip():
            logger.warning("Empty query provided")
            return None

        messages = [{'role': 'user', 'content': query}]
        iteration_count = 0

        try:
            while iteration_count < MAX_TOOL_CALL_ITERATIONS:
                iteration_count += 1
                logger.debug(f"Starting iteration {iteration_count}")

                # Get LLM response
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.available_tools,
                    tool_choice="auto",
                    max_tokens=self.max_tokens
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
                            # get tool executor from tool_to_session
                            tool_executor = self.tool_to_session[tool_name].call_tool

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

        logger.warning(f"Maximum iterations ({self.max_iterations}) reached")
        return None

    async def mcp_chat_loop(self):
        await chat_loop(
            query_executor=self.mcp_process_query
        )

    async def cleanup(self):
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()


async def main(server_config_path: str = "configs/mcp_server_configs.json"):
    chatbot = MCP_ChatBot(server_config_path)

    try:
        # the mcp clients and sessions are not initialized using "with"
        # like in the previous lesson
        # so the cleanup should be manually handled
        await chatbot.connect_to_servers()
        await chatbot.mcp_chat_loop()

    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    import fire
    fire.Fire(main)
