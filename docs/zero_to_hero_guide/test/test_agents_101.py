# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig

# Load environment variables
load_dotenv()

# Test configuration
HOST = os.getenv("LOCALHOST", "localhost")
PORT = os.getenv("PORT", "5000")
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


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
def travel_agent(client):
    """Create an agent with search capability for travel queries"""
    config = AgentConfig(
        model=MODEL_NAME,
        instructions="You are a helpful assistant! If you call builtin tools like brave search, follow the syntax brave_search.call(…)",
        sampling_params={
            "strategy": "greedy",
            "temperature": 1.0,
            "top_p": 0.9,
        },
        tools=[
            {
                "type": "brave_search",
                "engine": "brave",
                "api_key": os.environ.get("BRAVE_SEARCH_API_KEY", ""),
            }
        ],
        tool_choice="auto",
        tool_prompt_format="function_tag",
        input_shields=[],
        output_shields=[],
        enable_session_persistence=False,
    )
    return Agent(client, config)


@pytest.mark.asyncio
async def test_agent_initialization(travel_agent):
    """Test agent creation and session initialization"""
    session_id = travel_agent.create_session("test-session")
    assert session_id is not None
    assert travel_agent.agent_id is not None


@pytest.mark.asyncio
async def test_switzerland_query(travel_agent):
    """Test the Switzerland travel query from the notebook"""
    session_id = travel_agent.create_session("test-switzerland")

    # First query about Switzerland
    response = travel_agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": "I am planning a trip to Switzerland, what are the top 3 places to visit?",
            }
        ],
        session_id=session_id,
    )

    # Collect response and verify search was used
    search_used = False
    response_text = ""
    for log in EventLogger().log(response):
        log_str = str(log)
        response_text += log_str
        if "brave_search" in log_str:
            search_used = True

    # Verify search was used and got relevant results
    assert search_used, "Brave search should be used for this query"
    assert any(
        city.lower() in response_text.lower()
        for city in ["Zurich", "Geneva", "Bern", "Lucerne"]
    )


@pytest.mark.asyncio
async def test_follow_up_query(travel_agent):
    """Test the follow-up question about the first place"""
    session_id = travel_agent.create_session("test-follow-up")

    # First query to establish context
    response1 = travel_agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": "I am planning a trip to Switzerland, what are the top 3 places to visit?",
            }
        ],
        session_id=session_id,
    )

    # Consume first response
    first_response = ""
    for log in EventLogger().log(response1):
        first_response += str(log)

    # Follow-up question about the first place
    response2 = travel_agent.create_turn(
        messages=[{"role": "user", "content": "What is so special about #1?"}],
        session_id=session_id,
    )

    # Collect and verify follow-up response
    follow_up_text = ""
    for log in EventLogger().log(response2):
        follow_up_text += str(log)

    # Verify the follow-up response contains detailed information
    assert len(follow_up_text) > 0
    assert any(
        detail.lower() in follow_up_text.lower()
        for detail in ["attractions", "visit", "famous", "tourist"]
    )


@pytest.mark.asyncio
async def test_tool_functionality(travel_agent):
    """Test that the brave search tool works correctly"""
    session_id = travel_agent.create_session("test-tools")

    response = travel_agent.create_turn(
        messages=[
            {"role": "user", "content": "Find information about Swiss chocolate"}
        ],
        session_id=session_id,
    )

    # Verify tool usage and response
    tool_used = False
    response_text = ""
    for log in EventLogger().log(response):
        log_str = str(log)
        response_text += log_str
        if "brave_search" in log_str:
            tool_used = True

    assert tool_used, "Brave search tool should be used"
    assert "chocolate" in response_text.lower()


@pytest.mark.asyncio
async def test_error_handling(travel_agent):
    """Test error handling in agent interactions"""
    # Test with invalid session ID
    with pytest.raises(Exception):
        travel_agent.create_turn(
            messages=[{"role": "user", "content": "Hello"}],
            session_id="invalid_session_id",
        )

    # Test with empty messages
    with pytest.raises(Exception):
        travel_agent.create_turn(
            messages=[], session_id=travel_agent.create_session("error-session")
        )


if __name__ == "__main__":
    pytest.main(["-v", "test_agents_101.py"])
