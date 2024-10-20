# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, List, Optional, Protocol

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel, Field

from llama_stack.apis.common.type_system import ParamType


class Parameter(BaseModel):
    name: str
    type: ParamType
    description: Optional[str] = None


class ScoringFunctionDef(BaseModel):
    name: str = Field(
        description="A unique name for the scoring function",
    )
    parameters: List[Parameter] = Field(
        description="List of parameters for the scoring function",
    )
    return_type: ParamType = Field(
        description="The return type of the scoring function",
    )
    description: Optional[str] = Field(
        default=None,
        description="A description of what the scoring function does",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this scoring function",
    )

    # We can optionally add information here to support packaging of code, etc.


@json_schema_type
class ScoringFunctionDefWithProvider(ScoringFunctionDef):
    provider_id: str = Field(
        description="The provider ID for this scoring function",
    )


@runtime_checkable
class ScoringFunctions(Protocol):
    @webmethod(route="/scoring_functions/list", method="GET")
    async def list_scoring_functions(self) -> List[ScoringFunctionDefWithProvider]: ...

    @webmethod(route="/scoring_functions/get", method="GET")
    async def get_scoring_function(
        self, name: str
    ) -> Optional[ScoringFunctionDefWithProvider]: ...

    @webmethod(route="/scoring_functions/register", method="POST")
    async def register_scoring_function(
        self, function: ScoringFunctionDefWithProvider
    ) -> None: ...
