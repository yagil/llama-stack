# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.inference import Inference

from .config import LMStudioImplConfig

async def get_adapter_impl(config: LMStudioImplConfig, _deps) -> Inference:
    # import dynamically so `llama stack build` does not fail due to missing dependencies
    from .lmstudio import LMStudioInferenceAdapter

    if not isinstance(config, LMStudioImplConfig):
        raise RuntimeError(f"Unexpected config type: {type(config)}")
    adapter = LMStudioInferenceAdapter(config)
    return adapter


__all__ = ["get_adapter_impl", "LMStudioImplConfig"]
