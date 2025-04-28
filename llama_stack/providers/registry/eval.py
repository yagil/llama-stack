# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.providers.datatypes import AdapterSpec, Api, InlineProviderSpec, ProviderSpec, remote_provider_spec


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.eval,
            provider_type="inline::meta-reference",
            pip_packages=["tree_sitter", "pythainlp", "langdetect", "emoji", "nltk"],
            module="llama_stack.providers.inline.eval.meta_reference",
            config_class="llama_stack.providers.inline.eval.meta_reference.MetaReferenceEvalConfig",
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
                Api.scoring,
                Api.inference,
                Api.agents,
            ],
        ),
        remote_provider_spec(
            api=Api.eval,
            adapter=AdapterSpec(
                adapter_type="nvidia",
                pip_packages=[
                    "requests",
                ],
                module="llama_stack.providers.remote.eval.nvidia",
                config_class="llama_stack.providers.remote.eval.nvidia.NVIDIAEvalConfig",
            ),
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
                Api.scoring,
                Api.inference,
                Api.agents,
            ],
        ),
    ]
