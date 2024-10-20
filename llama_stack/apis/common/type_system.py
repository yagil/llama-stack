# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict, get_args, get_origin, List, Literal, Union

from pydantic import BaseModel, DiscriminatedUnion, Field
from typing_extensions import Annotated


class StringType(BaseModel):
    type: Literal["string"] = "string"


class NumberType(BaseModel):
    type: Literal["number"] = "number"


class BooleanType(BaseModel):
    type: Literal["boolean"] = "boolean"


class ArrayType(BaseModel):
    type: Literal["array"] = "array"
    items: "ParamType"


class ObjectType(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, "ParamType"] = Field(default_factory=dict)


class InferenceInputType(BaseModel):
    type: Literal["inference_input"] = "inference_input"


class InferenceOutputType(BaseModel):
    type: Literal["inference_output"] = "inference_output"


class TurnInputType(BaseModel):
    type: Literal["turn_input"] = "turn_input"


class TurnOutputType(BaseModel):
    type: Literal["turn_output"] = "turn_output"


class JsonType(BaseModel):
    type: Literal["json"] = "json"


class UnionType(BaseModel):
    type: Literal["union"] = "union"
    options: List["ParamType"] = Field(default_factory=list)


ParamType = Annotated[
    Union[
        StringType,
        NumberType,
        BooleanType,
        ArrayType,
        ObjectType,
        InferenceInputType,
        InferenceOutputType,
        TurnInputType,
        TurnOutputType,
        JsonType,
        UnionType,
    ],
    Field(discriminator="type"),
]


# TODO: move this to a pydantic util file
def get_discriminators_from_annotated(annotated_type) -> List[str]:
    args = get_args(annotated_type)
    if not args:
        return []

    union_type = args[0]
    if get_origin(union_type) is not Union:
        return []

    discriminator = next(
        (arg for arg in args[1:] if isinstance(arg, DiscriminatedUnion)), None
    )
    if not discriminator:
        return []

    return [
        getattr(subtype, discriminator.discriminator).default
        for subtype in get_args(union_type)
        if hasattr(subtype, discriminator.discriminator)
    ]


AllTypes = get_discriminators_from_annotated(ParamType)


ParamType.model_rebuild()
ArrayType.model_rebuild()
ObjectType.model_rebuild()
UnionType.model_rebuild()
