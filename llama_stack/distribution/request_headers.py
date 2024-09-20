# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import threading
from typing import Any, Dict, Optional

from .utils.dynamic import instantiate_class_type

_THREAD_LOCAL = threading.local()


def get_request_provider_data() -> Any:
    return getattr(_THREAD_LOCAL, "provider_data", None)


def set_request_provider_data(headers: Dict[str, str], class_type: Optional[str]):
    if not class_type:
        return

    val = headers.get("X-LlamaStack-ProviderData", None)
    if not val:
        return

    print("Got provider data", val)
    try:
        val = json.loads(val)
    except json.JSONDecodeError:
        print("Provider data not encoded as a JSON object!", val)
        return

    header_extractor = instantiate_class_type(class_type)
    try:
        provider_data = header_extractor(**val)
    except Exception as e:
        print("Error parsing provider data", e)
        return

    _THREAD_LOCAL.provider_data = provider_data
