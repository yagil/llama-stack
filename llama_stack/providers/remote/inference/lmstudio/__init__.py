# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import LMStudioImplConfig


async def get_adapter_impl(config: LMStudioImplConfig, _deps):
    from .lmstudio import LMStudioInferenceAdapter

    impl = LMStudioInferenceAdapter(config.url)
    await impl.initialize()
    return impl
