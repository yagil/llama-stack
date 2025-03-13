# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.models.llama.datatypes import CoreModelId

from llama_stack.apis.models.models import ModelType
from llama_stack.models.llama.datatypes import CoreModelId
from llama_stack.providers.utils.inference.model_registry import (
    ProviderModelEntry,
)

MODEL_ENTRIES = [
    ProviderModelEntry(
        provider_model_id="meta-llama3.1-8b-instruct",
        aliases=[],
        llama_model=CoreModelId.llama3_1_8b_instruct.value,
        model_type=ModelType.llm,
    ),
    ProviderModelEntry(
        provider_model_id="llama-3.2-3b-instruct",
        aliases=[],
        llama_model=CoreModelId.llama3_2_3b_instruct.value,
        model_type=ModelType.llm,
    ),
    # TODO add more models

    # embedding model
    ProviderModelEntry(
        model_id="nomic-embed-text-v1.5",
        provider_id="lmstudio",
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 768,
            "context_length": 2048,
        },
    )
]
