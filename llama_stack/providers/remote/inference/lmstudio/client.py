import asyncio
from typing import List, Literal, Optional
import lmstudio as lms

from llama_stack.apis.common.content_types import InterleavedContent
from llama_stack.apis.inference.inference import (
    ChatCompletionResponse,
    CompletionMessage,
    CompletionResponse,
    Message,
    ResponseFormat,
)
from llama_stack.models.llama.datatypes import (
    GreedySamplingStrategy,
    SamplingParams,
    StopReason,
    TopKSamplingStrategy,
    TopPSamplingStrategy,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    content_has_media,
    interleaved_content_as_str,
)

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


class LMStudioClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self.sdk_client = lms.Client(self.url)

    async def check_if_model_present_in_lmstudio(self, provider_model_id):
        models = await asyncio.to_thread(self.sdk_client.list_downloaded_models)
        model_ids = [m.model_key for m in models]
        model_ids = [m.model_key for m in models]
        if provider_model_id in model_ids:
            return True

        model_ids = [id.split("/")[-1] for id in model_ids]
        if provider_model_id in model_ids:
            return True
        return False

    async def get_embedding_model(self, provider_model_id: str):
        model = await asyncio.to_thread(
            self.sdk_client.embedding.model, provider_model_id
        )
        return model

    async def embed(
        self, embedding_model: lms.EmbeddingModel, contents: List[str] | List
    ):
        embeddings = await asyncio.to_thread(embedding_model.embed, contents)
        return embeddings

    async def get_llm(self, provider_model_id: str) -> lms.LLM:
        model = await asyncio.to_thread(self.sdk_client.llm.model, provider_model_id)
        return model

    async def llm_respond(
        self,
        llm: lms.LLM,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
    ) -> ChatCompletionResponse:
        chat = self._convert_message_list_to_lmstudio_chat(messages)
        config = self._get_completion_config_from_params(
            sampling_params, response_format
        )
        response = await asyncio.to_thread(llm.respond, history=chat, config=config)
        return self._convert_prediction_to_chat_response(response)

    async def llm_completion(
        self,
        llm: lms.LLM,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
    ) -> CompletionMessage:
        config = self._get_completion_config_from_params(
            sampling_params, response_format
        )
        response = await asyncio.to_thread(
            llm.complete, prompt=interleaved_content_as_str(content), config=config
        )
        return CompletionResponse(
            content=response.content,
            stop_reason=self._get_stop_reason(response.stats.stop_reason),
        )

    def _convert_message_list_to_lmstudio_chat(
        self, messages: List[Message]
    ) -> lms.Chat:
        chat = lms.Chat()
        for message in messages:
            if content_has_media(message.content):
                # TODO: Support images and other media
                continue
            if message.role == "user":
                chat.add_user_message(interleaved_content_as_str(message.content))
            elif message.role == "system":
                chat.add_system_prompt(interleaved_content_as_str(message.content))
            else:
                chat.add_assistant_response(interleaved_content_as_str(message.content))
        return chat

    def _convert_prediction_to_chat_response(
        self, result: lms.PredictionResult
    ) -> ChatCompletionResponse:
        response = ChatCompletionResponse(
            completion_message=CompletionMessage(
                content=result.content,
                stop_reason=self._get_stop_reason(result.stats.stop_reason),
                tool_calls=None,
            )
        )
        return response

    def _get_completion_config_from_params(
        self,
        params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
    ) -> lms.LlmPredictionConfigDict:
        options = lms.LlmPredictionConfigDict()
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
                    "maxTokens": params.max_tokens if params.max_tokens != 0 else None,
                    "repetitionPenalty": (
                        params.repetition_penalty
                        if params.repetition_penalty != 0
                        else None
                    ),
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
