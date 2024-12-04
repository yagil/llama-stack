# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os

import pytest
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.custom_tool import CustomTool
from llama_stack_client.types import CompletionMessage
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.shared.tool_response_message import ToolResponseMessage

# Load environment variables
load_dotenv()

# Test configuration
HOST = os.getenv("LOCALHOST", "localhost")
PORT = os.getenv("PORT", "5000")
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
BRAVE_SEARCH_API_KEY = os.environ.get("BRAVE_SEARCH_API_KEY", "")


class MockBraveSearch:
    """Mock BraveSearch class for testing"""

    async def search(self, query: str) -> str:
        mock_response = {
            "query": query,
            "top_k": [
                {
                    "title": "Test Result",
                    "url": "https://test.com",
                    "description": "Test description",
                }
            ],
        }
        return json.dumps(mock_response)


class TestWebSearchTool(CustomTool):
    """Test version of WebSearchTool using mock search"""

    def __init__(self):
        self.engine = MockBraveSearch()

    def get_name(self) -> str:
        return "web_search"

    def get_description(self) -> str:
        return "Search the web for a given query"

    async def run_impl(self, query: str):
        return await self.engine.search(query)

    async def run(self, messages):
        query = None
        call_id = None
        for message in messages:
            if isinstance(message, CompletionMessage) and message.tool_calls:
                for tool_call in message.tool_calls:
                    if "query" in tool_call.arguments:
                        query = tool_call.arguments["query"]
                        call_id = tool_call.call_id

        if query:
            search_result = await self.run_impl(query)
            return [
                ToolResponseMessage(
                    call_id=call_id,
                    role="ipython",
                    content=self._format_response_for_agent(search_result),
                    tool_name="web_search",
                )
            ]

        return [
            ToolResponseMessage(
                call_id="no_call_id",
                role="ipython",
                content="No query provided.",
                tool_name="web_search",
            )
        ]

    def _format_response_for_agent(self, search_result):
        parsed_result = json.loads(search_result)
        formatted_result = "Search Results with Citations:\n\n"
        for i, result in enumerate(parsed_result.get("top_k", []), start=1):
            formatted_result += (
                f"{i}. {result.get('title', 'No Title')}\n"
                f"   URL: {result.get('url', 'No URL')}\n"
                f"   Description: {result.get('description', 'No Description')}\n\n"
            )
        return formatted_result


@pytest.fixture
def client():
    """Create a reusable client fixture"""
    return LlamaStackClient(
        base_url=f"http://{HOST}:{PORT}",
        provider_data={
            "together_api_key": os.environ.get("TOGETHER_API_KEY", ""),
        },
    )


@pytest.fixture
def agent(client):
    """Create a reusable agent fixture"""
    agent_config = AgentConfig(
        model=MODEL_NAME,
        instructions="You are a helpful assistant that responds to user queries with relevant information.",
        sampling_params={
            "strategy": "greedy",
            "temperature": 1.0,
            "top_p": 0.9,
        },
        tools=[
            {
                "function_name": "web_search",
                "description": "Search the web for a given query",
                "parameters": {
                    "query": {
                        "param_type": "str",
                        "description": "The query to search for",
                        "required": True,
                    }
                },
                "type": "function_call",
            },
        ],
        tool_choice="auto",
        tool_prompt_format="python_list",
        input_shields=[],
        output_shields=[],
        enable_session_persistence=False,
    )

    custom_tools = [TestWebSearchTool()]
    return Agent(client, agent_config, custom_tools)


@pytest.mark.asyncio
async def test_web_search_tool_direct(client):
    """Test web search tool directly"""
    tool = TestWebSearchTool()
    result = await tool.run_impl("test query")

    assert result is not None
    parsed_result = json.loads(result)
    assert "query" in parsed_result
    assert "top_k" in parsed_result
    assert len(parsed_result["top_k"]) > 0


@pytest.mark.asyncio
async def test_agent_web_search(agent):
    """Test web search through agent"""
    session_id = agent.create_session("test-session")

    response = agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": "Search for information about quantum computing",
            }
        ],
        session_id=session_id,
    )

    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    assert len(chunks) > 0
    assert any("Search Results with Citations" in str(chunk) for chunk in chunks)


@pytest.mark.asyncio
async def test_tool_response_format(agent):
    """Test the format of tool responses"""
    tool = TestWebSearchTool()
    result = await tool.run_impl("test query")
    formatted = tool._format_response_for_agent(result)

    assert "Search Results with Citations:" in formatted
    assert "URL:" in formatted
    assert "Description:" in formatted


if __name__ == "__main__":
    pytest.main(["-v", "test_tool_calling_101.py"])
