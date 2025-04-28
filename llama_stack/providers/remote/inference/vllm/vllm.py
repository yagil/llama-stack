# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
import logging
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional, Union

import httpx
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk as OpenAIChatCompletionChunk,
)

from llama_stack.apis.common.content_types import (
    InterleavedContent,
    InterleavedContentItem,
    TextDelta,
    ToolCallDelta,
    ToolCallParseStatus,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamChunk,
    EmbeddingsResponse,
    EmbeddingTaskType,
    GrammarResponseFormat,
    Inference,
    JsonSchemaResponseFormat,
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
from llama_stack.apis.inference.inference import (
    OpenAIChatCompletion,
    OpenAICompletion,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
)
from llama_stack.apis.models import Model, ModelType
from llama_stack.models.llama.datatypes import BuiltinTool, StopReason, ToolCall
from llama_stack.models.llama.sku_list import all_registered_models
from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
    build_hf_repo_model_entry,
)
from llama_stack.providers.utils.inference.openai_compat import (
    UnparseableToolCall,
    convert_message_to_openai_dict,
    convert_tool_call,
    get_sampling_options,
    prepare_openai_completion_params,
    process_chat_completion_stream_response,
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    completion_request_to_prompt,
    content_has_media,
    interleaved_content_as_str,
    request_has_media,
)

from .config import VLLMInferenceAdapterConfig

log = logging.getLogger(__name__)


def build_hf_repo_model_entries():
    return [
        build_hf_repo_model_entry(
            model.huggingface_repo,
            model.descriptor(),
        )
        for model in all_registered_models()
        if model.huggingface_repo
    ]


def _convert_to_vllm_tool_calls_in_response(
    tool_calls,
) -> List[ToolCall]:
    if not tool_calls:
        return []

    return [
        ToolCall(
            call_id=call.id,
            tool_name=call.function.name,
            arguments=json.loads(call.function.arguments),
            arguments_json=call.function.arguments,
        )
        for call in tool_calls
    ]


def _convert_to_vllm_tools_in_request(tools: List[ToolDefinition]) -> List[dict]:
    compat_tools = []

    for tool in tools:
        properties = {}
        compat_required = []
        if tool.parameters:
            for tool_key, tool_param in tool.parameters.items():
                properties[tool_key] = {"type": tool_param.param_type}
                if tool_param.description:
                    properties[tool_key]["description"] = tool_param.description
                if tool_param.default:
                    properties[tool_key]["default"] = tool_param.default
                if tool_param.required:
                    compat_required.append(tool_key)

        # The tool.tool_name can be a str or a BuiltinTool enum. If
        # it's the latter, convert to a string.
        tool_name = tool.tool_name
        if isinstance(tool_name, BuiltinTool):
            tool_name = tool_name.value

        compat_tool = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": compat_required,
                },
            },
        }

        compat_tools.append(compat_tool)

    return compat_tools


def _convert_to_vllm_finish_reason(finish_reason: str) -> StopReason:
    return {
        "stop": StopReason.end_of_turn,
        "length": StopReason.out_of_tokens,
        "tool_calls": StopReason.end_of_message,
    }.get(finish_reason, StopReason.end_of_turn)


async def _process_vllm_chat_completion_stream_response(
    stream: AsyncGenerator[OpenAIChatCompletionChunk, None],
) -> AsyncGenerator:
    event_type = ChatCompletionResponseEventType.start
    tool_call_buf = UnparseableToolCall()
    async for chunk in stream:
        if not chunk.choices:
            log.warning("vLLM failed to generation any completions - check the vLLM server logs for an error.")
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            args_str = tool_call_buf.arguments
            args = None
            try:
                args = {} if not args_str else json.loads(args_str)
            except Exception as e:
                log.warning(f"Failed to parse tool call buffer arguments: {args_str} \nError: {e}")
            if args:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=event_type,
                        delta=ToolCallDelta(
                            tool_call=ToolCall(
                                call_id=tool_call_buf.call_id,
                                tool_name=tool_call_buf.tool_name,
                                arguments=args,
                                arguments_json=args_str,
                            ),
                            parse_status=ToolCallParseStatus.succeeded,
                        ),
                    )
                )
            elif args_str:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            tool_call=str(tool_call_buf),
                            parse_status=ToolCallParseStatus.failed,
                        ),
                    )
                )
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.complete,
                    delta=TextDelta(text=choice.delta.content or ""),
                    logprobs=None,
                    stop_reason=_convert_to_vllm_finish_reason(choice.finish_reason),
                )
            )
        elif choice.delta.tool_calls:
            tool_call = convert_tool_call(choice.delta.tool_calls[0])
            tool_call_buf.tool_name += str(tool_call.tool_name)
            tool_call_buf.call_id += tool_call.call_id
            # TODO: remove str() when dict type for 'arguments' is no longer allowed
            tool_call_buf.arguments += str(tool_call.arguments)
        else:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=event_type,
                    delta=TextDelta(text=choice.delta.content or ""),
                    logprobs=None,
                )
            )
            event_type = ChatCompletionResponseEventType.progress


class VLLMInferenceAdapter(Inference, ModelsProtocolPrivate):
    def __init__(self, config: VLLMInferenceAdapterConfig) -> None:
        self.register_helper = ModelRegistryHelper(build_hf_repo_model_entries())
        self.config = config
        self.client = None

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def _get_model(self, model_id: str) -> Model:
        if not self.model_store:
            raise ValueError("Model store not set")
        return await self.model_store.get_model(model_id)

    def _lazy_initialize_client(self):
        if self.client is not None:
            return

        log.info(f"Initializing vLLM client with base_url={self.config.url}")
        self.client = self._create_client()

    def _create_client(self):
        return AsyncOpenAI(
            base_url=self.config.url,
            api_key=self.config.api_token,
            http_client=None if self.config.tls_verify else httpx.AsyncClient(verify=False),
        )

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> CompletionResponse | AsyncGenerator[CompletionResponseStreamChunk, None]:
        self._lazy_initialize_client()
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self._get_model(model_id)
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
    ) -> ChatCompletionResponse | AsyncGenerator[ChatCompletionResponseStreamChunk, None]:
        self._lazy_initialize_client()
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self._get_model(model_id)
        # This is to be consistent with OpenAI API and support vLLM <= v0.6.3
        # References:
        #   * https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
        #   * https://github.com/vllm-project/vllm/pull/10000
        if not tools and tool_config is not None:
            tool_config.tool_choice = ToolChoice.none
        request = ChatCompletionRequest(
            model=model.provider_resource_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            stream=stream,
            logprobs=logprobs,
            response_format=response_format,
            tool_config=tool_config,
        )
        if stream:
            return self._stream_chat_completion(request, self.client)
        else:
            return await self._nonstream_chat_completion(request, self.client)

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest, client: AsyncOpenAI
    ) -> ChatCompletionResponse:
        params = await self._get_params(request)
        r = await client.chat.completions.create(**params)
        choice = r.choices[0]
        result = ChatCompletionResponse(
            completion_message=CompletionMessage(
                content=choice.message.content or "",
                stop_reason=_convert_to_vllm_finish_reason(choice.finish_reason),
                tool_calls=_convert_to_vllm_tool_calls_in_response(choice.message.tool_calls),
            ),
            logprobs=None,
        )
        return result

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest, client: AsyncOpenAI
    ) -> AsyncGenerator[ChatCompletionResponseStreamChunk, None]:
        params = await self._get_params(request)

        stream = await client.chat.completions.create(**params)
        if request.tools:
            res = _process_vllm_chat_completion_stream_response(stream)
        else:
            res = process_chat_completion_stream_response(stream, request)
        async for chunk in res:
            yield chunk

    async def _nonstream_completion(self, request: CompletionRequest) -> CompletionResponse:
        assert self.client is not None
        params = await self._get_params(request)
        r = await self.client.completions.create(**params)
        return process_completion_response(r)

    async def _stream_completion(
        self, request: CompletionRequest
    ) -> AsyncGenerator[CompletionResponseStreamChunk, None]:
        assert self.client is not None
        params = await self._get_params(request)

        stream = await self.client.completions.create(**params)
        async for chunk in process_completion_stream_response(stream):
            yield chunk

    async def register_model(self, model: Model) -> Model:
        # register_model is called during Llama Stack initialization, hence we cannot init self.client if not initialized yet.
        # self.client should only be created after the initialization is complete to avoid asyncio cross-context errors.
        # Changing this may lead to unpredictable behavior.
        client = self._create_client() if self.client is None else self.client
        model = await self.register_helper.register_model(model)
        res = await client.models.list()
        available_models = [m.id async for m in res]
        if model.provider_resource_id not in available_models:
            raise ValueError(
                f"Model {model.provider_resource_id} is not being served by vLLM. "
                f"Available models: {', '.join(available_models)}"
            )
        return model

    async def _get_params(self, request: Union[ChatCompletionRequest, CompletionRequest]) -> dict:
        options = get_sampling_options(request.sampling_params)
        if "max_tokens" not in options:
            options["max_tokens"] = self.config.max_tokens

        input_dict: dict[str, Any] = {}
        # Only include the 'tools' param if there is any. It can break things if an empty list is sent to the vLLM.
        if isinstance(request, ChatCompletionRequest) and request.tools:
            input_dict = {"tools": _convert_to_vllm_tools_in_request(request.tools)}

        if isinstance(request, ChatCompletionRequest):
            input_dict["messages"] = [await convert_message_to_openai_dict(m, download=True) for m in request.messages]
        else:
            assert not request_has_media(request), "vLLM does not support media for Completion requests"
            input_dict["prompt"] = await completion_request_to_prompt(request)

        if fmt := request.response_format:
            if isinstance(fmt, JsonSchemaResponseFormat):
                input_dict["extra_body"] = {"guided_json": fmt.json_schema}
            elif isinstance(fmt, GrammarResponseFormat):
                raise NotImplementedError("Grammar response format not supported yet")
            else:
                raise ValueError(f"Unknown response format {fmt.type}")

        if request.logprobs and request.logprobs.top_k:
            input_dict["logprobs"] = request.logprobs.top_k

        return {
            "model": request.model,
            **input_dict,
            "stream": request.stream,
            **options,
        }

    async def embeddings(
        self,
        model_id: str,
        contents: List[str] | List[InterleavedContentItem],
        text_truncation: Optional[TextTruncation] = TextTruncation.none,
        output_dimension: Optional[int] = None,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        self._lazy_initialize_client()
        assert self.client is not None
        model = await self._get_model(model_id)

        kwargs = {}
        assert model.model_type == ModelType.embedding
        assert model.metadata.get("embedding_dimension")
        kwargs["dimensions"] = model.metadata.get("embedding_dimension")
        assert all(not content_has_media(content) for content in contents), "VLLM does not support media for embeddings"
        response = await self.client.embeddings.create(
            model=model.provider_resource_id,
            input=[interleaved_content_as_str(content) for content in contents],
            **kwargs,
        )

        embeddings = [data.embedding for data in response.data]
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
        self._lazy_initialize_client()
        model_obj = await self._get_model(model)

        extra_body: Dict[str, Any] = {}
        if prompt_logprobs is not None and prompt_logprobs >= 0:
            extra_body["prompt_logprobs"] = prompt_logprobs
        if guided_choice:
            extra_body["guided_choice"] = guided_choice

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
            extra_body=extra_body,
        )
        return await self.client.completions.create(**params)  # type: ignore

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
        self._lazy_initialize_client()
        model_obj = await self._get_model(model)
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
        return await self.client.chat.completions.create(**params)  # type: ignore

    async def batch_completion(
        self,
        model_id: str,
        content_batch: List[InterleavedContent],
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        logprobs: Optional[LogProbConfig] = None,
    ):
        raise NotImplementedError("Batch completion is not supported for Ollama")

    async def batch_chat_completion(
        self,
        model_id: str,
        messages_batch: List[List[Message]],
        sampling_params: Optional[SamplingParams] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_config: Optional[ToolConfig] = None,
        response_format: Optional[ResponseFormat] = None,
        logprobs: Optional[LogProbConfig] = None,
    ):
        raise NotImplementedError("Batch chat completion is not supported for Ollama")
