from pydantic import Field
from typing_extensions import Annotated

import threading

_thread_local = threading.local()

def set_header_context(context: Any) -> None:
    _thread_local.context = context

def get_header_context() -> Any:
    return getattr(_thread_local, 'context', None)


def annotate_header(
    typ: Type, header_name: str, description: str
):
    assert header_name.startswith("X-LlamaStack-")
    return Annotated[
        typ,
        Field(
            description=description,
            alias=header_name.replace("-", "_"),
            default=None,
        ),
    ]
