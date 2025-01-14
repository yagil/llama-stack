# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# import os

from llama_stack.distribution.library_client import LlamaStackAsLibraryClient

# from llama_stack_client.lib.inference.event_logger import EventLogger

# os.environ["INFERENCE_MODEL"] = "meta-llama/Llama-3.1-8B-Instruct"
# os.environ["OLLAMA_INFERENCE_MODEL"] = "llama3.1:8b-instruct-fp16"
client = LlamaStackAsLibraryClient("fireworks", skip_logger_removal=True)
_ = client.initialize()

model_id = "meta-llama/Llama-3.1-8B-Instruct"

response = client.inference.chat_completion(
    model_id=model_id,
    messages=[
        {"role": "system", "content": "You are a friendly assistant."},
        {"role": "user", "content": "Write a two-sentence poem about llama."},
    ],
    stream=True,
)
for x in response:
    print(x)
# for log in EventLogger().log(response):
#     log.print()


# response2 = client.inference.chat_completion(
#     model_id=model_id,
#     messages=[
#         {"role": "user", "content": "What's up?"},
#     ],
#     stream=True,
# )
# for log in EventLogger().log(response2):
#     log.print()
