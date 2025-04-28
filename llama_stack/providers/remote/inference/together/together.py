# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional, Union

from openai import AsyncOpenAI
from together import AsyncTogether

from llama_stack.apis.common.content_types import (
    InterleavedContent,
    InterleavedContentItem,
)
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
    ResponseFormatType,
    SamplingParams,
    TextTruncation,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.apis.inference.inference import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAICompletion,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
)
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_compat import (
    convert_message_to_openai_dict,
    get_sampling_options,
    prepare_openai_completion_params,
    process_chat_completion_response,
    process_chat_completion_stream_response,
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
    completion_request_to_prompt,
    content_has_media,
    interleaved_content_as_str,
    request_has_media,
)

from .config import TogetherImplConfig
from .models import MODEL_ENTRIES

logger = get_logger(name=__name__, category="inference")


class TogetherInferenceAdapter(ModelRegistryHelper, Inference, NeedsRequestProviderData):
    def __init__(self, config: TogetherImplConfig) -> None:
        ModelRegistryHelper.__init__(self, MODEL_ENTRIES)
        self.config = config
        self._client = None
        self._openai_client = None

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        if self._client:
            # Together client has no close method, so just set to None
            self._client = None
        if self._openai_client:
            await self._openai_client.close()
            self._openai_client = None

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self.model_store.get_model(model_id)
        request = CompletionRequest(
            model=model.provider_resource_id,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        if stream:
            return self._stream_completion(request)
        else:
            return await self._nonstream_completion(request)

    def _get_client(self) -> AsyncTogether:
        if not self._client:
            together_api_key = None
            config_api_key = self.config.api_key.get_secret_value() if self.config.api_key else None
            if config_api_key:
                together_api_key = config_api_key
            else:
                provider_data = self.get_request_provider_data()
                if provider_data is None or not provider_data.together_api_key:
                    raise ValueError(
                        'Pass Together API Key in the header X-LlamaStack-Provider-Data as { "together_api_key": <your api key>}'
                    )
                together_api_key = provider_data.together_api_key
            self._client = AsyncTogether(api_key=together_api_key)
        return self._client

    def _get_openai_client(self) -> AsyncOpenAI:
        if not self._openai_client:
            together_client = self._get_client().client
            self._openai_client = AsyncOpenAI(
                base_url=together_client.base_url,
                api_key=together_client.api_key,
            )
        return self._openai_client

    async def _nonstream_completion(self, request: CompletionRequest) -> ChatCompletionResponse:
        params = await self._get_params(request)
        client = self._get_client()
        r = await client.completions.create(**params)
        return process_completion_response(r)

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)
        client = self._get_client()
        stream = await client.completions.create(**params)
        async for chunk in process_completion_stream_response(stream):
            yield chunk

    def _build_options(
        self,
        sampling_params: Optional[SamplingParams],
        logprobs: Optional[LogProbConfig],
        fmt: ResponseFormat,
    ) -> dict:
        options = get_sampling_options(sampling_params)
        if fmt:
            if fmt.type == ResponseFormatType.json_schema.value:
                options["response_format"] = {
                    "type": "json_object",
                    "schema": fmt.json_schema,
                }
            elif fmt.type == ResponseFormatType.grammar.value:
                raise NotImplementedError("Grammar response format not supported yet")
            else:
                raise ValueError(f"Unknown response format {fmt.type}")

        if logprobs and logprobs.top_k:
            if logprobs.top_k != 1:
                raise ValueError(
                    f"Unsupported value: Together only supports logprobs top_k=1. {logprobs.top_k} was provided",
                )
            options["logprobs"] = 1

        return options

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self.model_store.get_model(model_id)
        request = ChatCompletionRequest(
            model=model.provider_resource_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
            tool_config=tool_config,
        )

        if stream:
            return self._stream_chat_completion(request)
        else:
            return await self._nonstream_chat_completion(request)

    async def _nonstream_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        params = await self._get_params(request)
        client = self._get_client()
        if "messages" in params:
            r = await client.chat.completions.create(**params)
        else:
            r = await client.completions.create(**params)
        return process_chat_completion_response(r, request)

    async def _stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)
        client = self._get_client()
        if "messages" in params:
            stream = await client.chat.completions.create(**params)
        else:
            stream = await client.completions.create(**params)

        async for chunk in process_chat_completion_stream_response(stream, request):
            yield chunk

    async def _get_params(self, request: Union[ChatCompletionRequest, CompletionRequest]) -> dict:
        input_dict = {}
        media_present = request_has_media(request)
        llama_model = self.get_llama_model(request.model)
        if isinstance(request, ChatCompletionRequest):
            if media_present or not llama_model:
                input_dict["messages"] = [await convert_message_to_openai_dict(m) for m in request.messages]
            else:
                input_dict["prompt"] = await chat_completion_request_to_prompt(request, llama_model)
        else:
            assert not media_present, "Together does not support media for Completion requests"
            input_dict["prompt"] = await completion_request_to_prompt(request)

        params = {
            "model": request.model,
            **input_dict,
            "stream": request.stream,
            **self._build_options(request.sampling_params, request.logprobs, request.response_format),
        }
        logger.debug(f"params to together: {params}")
        return params

    async def embeddings(
        self,
        model_id: str,
        contents: List[str] | List[InterleavedContentItem],
        text_truncation: Optional[TextTruncation] = TextTruncation.none,
        output_dimension: Optional[int] = None,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        model = await self.model_store.get_model(model_id)
        assert all(not content_has_media(content) for content in contents), (
            "Together does not support media for embeddings"
        )
        client = self._get_client()
        r = await client.embeddings.create(
            model=model.provider_resource_id,
            input=[interleaved_content_as_str(content) for content in contents],
        )
        embeddings = [item.embedding for item in r.data]
        return EmbeddingsResponse(embeddings=embeddings)

    async def openai_completion(
        self,
        model: str,
        prompt: Union[str, List[str], List[int], List[List[int]]],
        best_of: Optional[int] = None,
        echo: Optional[bool] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        prompt_logprobs: Optional[int] = None,
    ) -> OpenAICompletion:
        model_obj = await self.model_store.get_model(model)
        params = await prepare_openai_completion_params(
            model=model_obj.provider_resource_id,
            prompt=prompt,
            best_of=best_of,
            echo=echo,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            top_p=top_p,
            user=user,
        )
        return await self._get_openai_client().completions.create(**params)  # type: ignore

    async def openai_chat_completion(
        self,
        model: str,
        messages: List[OpenAIMessageParam],
        frequency_penalty: Optional[float] = None,
        function_call: Optional[Union[str, Dict[str, Any]]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[OpenAIResponseFormatParam] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
    ) -> Union[OpenAIChatCompletion, AsyncIterator[OpenAIChatCompletionChunk]]:
        model_obj = await self.model_store.get_model(model)
        params = await prepare_openai_completion_params(
            model=model_obj.provider_resource_id,
            messages=messages,
            frequency_penalty=frequency_penalty,
            function_call=function_call,
            functions=functions,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            n=n,
            parallel_tool_calls=parallel_tool_calls,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=tools,
            top_logprobs=top_logprobs,
            top_p=top_p,
            user=user,
        )
        if params.get("stream", False):
            return self._stream_openai_chat_completion(params)
        return await self._get_openai_client().chat.completions.create(**params)  # type: ignore

    async def _stream_openai_chat_completion(self, params: dict) -> AsyncGenerator:
        # together.ai sometimes adds usage data to the stream, even if include_usage is False
        # This causes an unexpected final chunk with empty choices array to be sent
        # to clients that may not handle it gracefully.
        include_usage = False
        if params.get("stream_options", None):
            include_usage = params["stream_options"].get("include_usage", False)
        stream = await self._get_openai_client().chat.completions.create(**params)

        seen_finish_reason = False
        async for chunk in stream:
            # Final usage chunk with no choices that the user didn't request, so discard
            if not include_usage and seen_finish_reason and len(chunk.choices) == 0:
                break
            yield chunk
            for choice in chunk.choices:
                if choice.finish_reason:
                    seen_finish_reason = True
                    break
