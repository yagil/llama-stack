# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import threading
from typing import Any, Type

from pydantic import Field
from typing_extensions import Annotated

_THREAD_LOCAL = threading.local()


def set_request_provider_data(provider_data: Any) -> None:
    _THREAD_LOCAL.provider_data = provider_data


def get_request_provider_data() -> Any:
    return getattr(_THREAD_LOCAL, "provider_data", None)


def annotate_header(typ: Type, header_name: str, description: str):
    assert header_name.startswith("X-LlamaStack-")
    return Annotated[
        typ,
        Field(
            description=description,
            alias=header_name.replace("-", "_"),
            default=None,
        ),
    ]
