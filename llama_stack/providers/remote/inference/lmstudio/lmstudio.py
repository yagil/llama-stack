import lmstudio as lms
from lmstudio import LlmPredictionConfigDict
from llama_stack.apis.common.content_types import InterleavedContentItem
from llama_stack.apis.inference.inference import (
    ChatCompletionResponseStreamChunk,
    CompletionResponse,
    CompletionResponseStreamChunk,
)
from llama_stack.models.llama.datatypes import (
    GreedySamplingStrategy,
    StopReason,
    TopKSamplingStrategy,
    TopPSamplingStrategy,
)
from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.apis.inference import Inference
from typing import AsyncGenerator, AsyncIterator, List, Literal, Optional, Union
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
    InterleavedContent,
    InterleavedContentItem,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    content_has_media,
    interleaved_content_as_str,
)

from .models import MODEL_ENTRIES
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
import asyncio
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="inference")


LlmPredictionStopReason = Literal[
    "userStopped",
    "modelUnloaded",
    "failed",
    "eosFound",
    "stopStringFound",
    "toolCalls",
    "maxPredictedTokensReached",
    "contextLengthReached",
]


class LmstudioInferenceAdapter(Inference, ModelsProtocolPrivate):
    def __init__(self, url: str) -> None:
        self.url = url
        self.client = None
        self.register_helper = ModelRegistryHelper(MODEL_ENTRIES)

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
        if model.provider_model_id in model_ids:
            return model

        model_ids = [id.split("/")[-1] for id in model_ids]
        if model.provider_model_id in model_ids:
            return model

        raise ValueError(f"Model {model.provider_model_id} not found in LM Studio")

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
        model = await asyncio.to_thread(
            self.client.embedding.model, embedding_model.provider_model_id
        )

        embeddings = await asyncio.to_thread(
            model.embed, interleaved_content_as_str(contents)
        )

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

        llm: lms.LLM = await asyncio.to_thread(self.client.llm.model(model_id))
        text_content = [message.content.text for message in messages]
        res = await asyncio.to_thread(llm.respond, text_content)
        return res

    def _get_completion_config_from_params(
        self,
        params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
    ) -> LlmPredictionConfigDict:
        options = LlmPredictionConfigDict()
        if params is not None:
            if isinstance(params.strategy, GreedySamplingStrategy):
                options.update({"temperature": 0.0})
            elif isinstance(params.strategy, TopPSamplingStrategy):
                options.update(
                    {
                        "temperature": params.strategy.temperature,
                        "top_p": params.strategy.top_p,
                    }
                )
            elif isinstance(params.strategy, TopKSamplingStrategy):
                options.update({"topKSampling": params.strategy.top_k})
            else:
                raise ValueError(f"Unsupported sampling strategy: {params.strategy}")
            options.update(
                {
                    "maxTokens": params.max_tokens,
                    "repetitionPenalty": params.repetition_penalty,
                }
            )
        if response_format is not None:
            if response_format.type == "json_schema":
                options.update(
                    {
                        "structured": {
                            "type": "json",
                            "jsonSchema": response_format.json_schema,
                        }
                    }
                )
            elif response_format.type == "grammar":
                raise NotImplementedError("Grammar response format is not supported")
            else:
                raise ValueError(f"Unsupported response format: {response_format}")
        return options

    def _get_stop_reason(self, stop_reason: LlmPredictionStopReason) -> StopReason:
        if stop_reason == "eosFound":
            return StopReason.end_of_message
        elif stop_reason == "maxPredictedTokensReached":
            return StopReason.out_of_tokens
        else:
            return StopReason.end_of_turn

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,  # Skip this for now
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponseStreamChunk]]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self.model_store.get_model(model_id)
        config = self._get_completion_config_from_params(
            sampling_params, response_format
        )
        llm = await asyncio.to_thread(self.client.llm.model, model.provider_model_id)
        if stream:
            pass
        else:
            response = await asyncio.to_thread(
                llm.complete, prompt=interleaved_content_as_str(content), config=config
            )
            return CompletionResponse(
                content=response.content,
                stop_reason=self._get_stop_reason(response.stats.stop_reason),
            )
