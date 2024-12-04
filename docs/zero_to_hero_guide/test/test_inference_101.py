# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.inference.event_logger import EventLogger

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


@pytest.mark.asyncio
async def test_basic_chat_completion(client):
    """Test basic chat completion functionality"""
    response = client.inference.chat_completion(
        messages=[
            {"role": "system", "content": "You are a friendly assistant."},
            {"role": "user", "content": "Write a two-sentence poem about llama."},
        ],
        model_id=MODEL_NAME,
    )

    assert response.completion_message.content is not None
    assert len(response.completion_message.content.split(".")) >= 2
    assert "llama" in response.completion_message.content.lower()


@pytest.mark.asyncio
async def test_shakespeare_style_completion(client):
    """Test chat completion with Shakespeare persona"""
    response = client.inference.chat_completion(
        messages=[
            {"role": "system", "content": "You are shakespeare."},
            {"role": "user", "content": "Write a two-sentence poem about llama."},
        ],
        model_id=MODEL_NAME,
    )

    assert response.completion_message.content is not None
    assert len(response.completion_message.content.split(".")) >= 2
    assert "llama" in response.completion_message.content.lower()


@pytest.mark.asyncio
async def test_streaming_response(client):
    """Test streaming response functionality"""
    response = client.inference.chat_completion(
        messages=[{"role": "user", "content": "Write a 3 sentence poem about llama"}],
        model_id=MODEL_NAME,
        stream=True,
    )

    # Collect all streamed chunks
    chunks = []
    async for log in EventLogger().log(response):
        chunks.append(str(log))

    # Join all chunks
    complete_response = "".join(chunks)

    assert len(complete_response) > 0
    assert "llama" in complete_response.lower()
    assert len(complete_response.split(".")) >= 3


@pytest.mark.asyncio
async def test_error_handling(client):
    """Test error handling with invalid inputs"""
    with pytest.raises(Exception):
        await client.inference.chat_completion(
            messages=[],  # Empty messages should raise an error
            model_id=MODEL_NAME,
        )

    with pytest.raises(Exception):
        await client.inference.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model_id="invalid-model-name",  # Invalid model name should raise an error
        )


if __name__ == "__main__":
    pytest.main(["-v", "test_inference_101.py"])
