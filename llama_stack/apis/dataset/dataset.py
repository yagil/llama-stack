# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from llama_models.llama3.api.datatypes import URL
from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel, Field


class ColumnType(str, Enum):
    # Primitive types
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"

    # Special server-validated types
    INFERENCE_INPUT = "inference_input"
    INFERENCE_OUTPUT = "inference_output"
    TURN_INPUT = "turn_input"
    TURN_OUTPUT = "turn_output"

    # Blob
    JSON = "json"


class ListType(BaseModel):
    type: Literal["array"] = "array"
    items: Union[ColumnType, "ComplexType"]


class DictType(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, Union[ColumnType, "ComplexType"]] = Field(
        default_factory=dict
    )


class ComplexType(BaseModel):
    type: Union[ListType, DictType]


ComplexType.model_rebuild()
ListType.model_rebuild()

ColumnDefinition = Union[ColumnType, ComplexType]


class DatasetSchema(BaseModel):
    columns: Dict[str, ColumnDefinition]


class DatasetDef(BaseModel):
    identifier: str = Field(
        description="A unique name for the dataset",
    )
    schema: DatasetSchema = Field(
        description="The schema definition for this dataset",
    )
    url: URL
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this dataset",
    )


@json_schema_type
class DatasetDefWithProvider(DatasetDef):
    provider_id: str = Field(
        description="ID of the provider which serves this dataset",
    )


@runtime_checkable
class Datasets(Protocol):
    @webmethod(route="/datasets/list", method="GET")
    async def list_datasets(self) -> List[DatasetDefWithProvider]: ...

    @webmethod(route="/datasets/get", method="GET")
    async def get_dataset(
        self, identifier: str
    ) -> Optional[DatasetDefWithProvider]: ...

    @webmethod(route="/datasets/register", method="POST")
    async def register_dataset(self, model: DatasetDefWithProvider) -> None: ...
