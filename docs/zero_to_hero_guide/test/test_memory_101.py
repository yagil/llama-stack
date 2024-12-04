# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import mimetypes
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.memory_insert_params import Document

# Load environment variables
load_dotenv()

# Test configuration
HOST = os.getenv("LOCALHOST", "localhost")
PORT = os.getenv("PORT", "5000")
TEST_BANK_ID = "test_memory_bank"


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
def test_documents():
    """Create test documents for memory operations"""
    return [
        Document(
            document_id="test-doc-1",
            content="This is a test document about llamas. Llamas are camelids native to South America.",
            mime_type="text/plain",
            metadata={"source": "test", "category": "animals"},
        ),
        Document(
            document_id="test-doc-2",
            content="AI and machine learning are transforming technology. Neural networks process data efficiently.",
            mime_type="text/plain",
            metadata={"source": "test", "category": "technology"},
        ),
    ]


def data_url_from_content(content: str) -> str:
    """Helper function to create data URL from text content"""
    encoded = base64.b64encode(content.encode()).decode()
    return f"data:text/plain;base64,{encoded}"


@pytest.mark.asyncio
async def test_memory_bank_creation(client):
    """Test creating a memory bank"""
    # Register a new memory bank
    response = client.memory_banks.register(
        memory_bank_id=TEST_BANK_ID,
        params={
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size_in_tokens": 512,
            "overlap_size_in_tokens": 64,
        },
        provider_id="faiss",
    )

    assert response is not None

    # Verify bank exists in list
    banks = client.memory_banks.list()
    assert TEST_BANK_ID in [bank.bank_id for bank in banks]


@pytest.mark.asyncio
async def test_document_insertion(client, test_documents):
    """Test inserting documents into memory bank"""
    response = client.memory.insert(bank_id=TEST_BANK_ID, documents=test_documents)

    assert response is not None
    assert len(response.document_ids) == len(test_documents)


@pytest.mark.asyncio
async def test_memory_query(client):
    """Test querying the memory bank"""
    query = "What are llamas?"
    response = client.memory.query(bank_id=TEST_BANK_ID, query=[query])

    assert response.chunks is not None
    assert len(response.chunks) > 0
    assert response.scores is not None
    assert len(response.scores) > 0

    # Check if the most relevant chunk contains related content
    most_relevant = response.chunks[0]
    assert "llama" in most_relevant.content.lower()


@pytest.mark.asyncio
async def test_metadata_filtering(client, test_documents):
    """Test querying with metadata filters"""
    query = "Tell me about technology"
    response = client.memory.query(
        bank_id=TEST_BANK_ID, query=[query], filter={"category": "technology"}
    )

    assert response.chunks is not None
    assert len(response.chunks) > 0
    # Verify the returned chunk is from the technology category
    assert any("neural networks" in chunk.content.lower() for chunk in response.chunks)


@pytest.mark.asyncio
async def test_memory_bank_deletion(client):
    """Test deleting a memory bank"""
    response = client.memory_banks.delete(TEST_BANK_ID)
    assert response is not None

    # Verify bank is removed from list
    banks = client.memory_banks.list()
    assert TEST_BANK_ID not in [bank.bank_id for bank in banks]


@pytest.mark.asyncio
async def test_error_handling(client):
    """Test error handling for invalid operations"""
    # Test querying non-existent bank
    with pytest.raises(Exception):
        await client.memory.query(bank_id="nonexistent-bank", query=["test query"])

    # Test inserting with invalid document format
    with pytest.raises(Exception):
        await client.memory.insert(
            bank_id=TEST_BANK_ID, documents=[{"invalid": "document"}]
        )


if __name__ == "__main__":
    pytest.main(["-v", "test_memory_101.py"])
