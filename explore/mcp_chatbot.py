import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import List

from llm_tool_use import process_query, chat_loop


load_dotenv()

llm_client = AsyncOpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url=os.getenv("GOOGLE_OPENAI_API_ENDPOINT")
)


class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.session: ClientSession = None
        self.client = llm_client
        self.available_tools: List[dict] = []

    async def mcp_process_query(self, query):
        return await process_query(
            query=query,
            client=self.client,
            tools=self.available_tools,
            tool_executor=self.session.call_tool
        )

    async def mcp_chat_loop(self):
        await chat_loop(
            query_executor=self.mcp_process_query
        )

    async def connect_to_server_and_run(self):
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="python",  # Executable
            args=["explore/tools/arxiv_research.py"],  # Command line arguments
            env=None,  # Optional environment variables
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                # Initialize the connection
                await session.initialize()

                # List available tools
                response = await session.list_tools()

                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])
                # import pdb; pdb.set_trace()
                self.available_tools = [{
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                } for tool in response.tools]

                await self.mcp_chat_loop()


async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":
    asyncio.run(main())
