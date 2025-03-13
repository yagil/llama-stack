import lmstudio as lms
from lmstudio import SyncSessionEmbedding
from llama_stack.apis.common.content_types import InterleavedContentItem
from llama_stack.apis.inference.inference import (
    ChatCompletionResponseStreamChunk,
    CompletionResponse,
    CompletionResponseStreamChunk,
)
from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.apis.inference import Inference
from typing import AsyncGenerator, AsyncIterator, List, Optional, Union
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    EmbeddingsResponse,
    EmbeddingTaskType,
    Inference,
    LogProbConfig,
    Message,
    ResponseFormat,
    SamplingParams,
    TextTruncation,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.apis.common.content_types import (
    ImageContentItem,
    InterleavedContent,
    InterleavedContentItem,
    TextContentItem,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
    completion_request_to_prompt,
    content_has_media,
    convert_image_content_to_url,
    interleaved_content_as_str,
    request_has_media,
)

from .models import model_entries
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
import asyncio
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="inference")


class LMStudioInferenceAdapter(Inference, ModelsProtocolPrivate):
    def __init__(self, url: str) -> None:
        self.url = url
        self.client = None
        self.register_helper = ModelRegistryHelper(model_entries)

    def _get_client(self) -> lms.Client:
        if self.client is None:
            self.client = lms.Client(self.url)
        return self.client

    async def initialize(self) -> None:
        self.client = self._get_client()

    async def register_model(self, model):
        model = await self.register_helper.register_model(model)
        models = await asyncio.to_thread(self.client.list_downloaded_models)
        model_ids = [m.model_key for m in models]
        if model.provider_model_id not in model_ids:
            raise ValueError(f"Model {model.provider_model_id} not found in LM Studio")
        return model

    async def unregister_model(self, model_id):
        pass

    async def embeddings(
        self,
        model_id: str,
        contents: List[str] | List[InterleavedContentItem],
        text_truncation: Optional[TextTruncation] = TextTruncation.none,
        output_dimension: Optional[int] = None,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        model = self.model_store.get_model(model_id)

        assert all(
            not content_has_media(content) for content in contents
        ), "Media content not supported in embedding model"
        embedding_model = self.model_store.get_model(model_id)
        model = await asyncio.to_thread(self.client.embedding.model, embedding_model.provider_model_id)

        embeddings = await asyncio.to_thread(model.embed, interleaved_content_as_str(contents))

        return EmbeddingsResponse(embeddings=embeddings)

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> Union[
        ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]
    ]:

        llm: lms.LLM = await asyncio.to_thread(
            self.client.llm.load_new_instance(model_id)
        )
        text_content = [message.content.text for message in messages]
        res = await asyncio.to_thread(llm.respond(text_content))
        return res

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponseStreamChunk]]:

        llm: lms.LLM = await asyncio.to_thread(
            self.client.llm.load_new_instance(model_id)
        )
        res = await asyncio.to_thread(llm.respond(content.text))
        print(res)
        return res
