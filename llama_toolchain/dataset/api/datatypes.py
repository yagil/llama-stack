# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import Annotated

from llama_models.llama3_1.api.datatypes import URL

from llama_models.schema_utils import json_schema_type

from pydantic import BaseModel


@json_schema_type
class ColumnType(Enum):
    row_id = "row_id"

    input_question = "input_question"
    input_messages = "input_messages"
    output_completion = "output_completion"

    answers = "answers"
    choices = "choices"

    chosen_response = "chosen_response"
    rejected_response = "rejected_response"


class BaseDataset(BaseModel):
    """Dataset to be used for training or evaluating language models."""
    name: str
    description: str
    url: URL
    mime_type: str
    column_map: Dict[str, ColumnType]
    metadata: Optional[Dict[str, Any]] = None


class DatasetType(Enum):
    completion = "completion"
    instruct = "instruct"
    chat = "chat"
    preference = "preference"
    evaluation = "evaluation"
    eval_generations = "eval_generations"


# NOTE: at this point, there are no fields specific to each dataset type
# but it seems reasonable to make space for them in the future.

@json_schema_type
class CompletionDataset(BaseDataset):
    type: Literal[DatasetType.completion.value] = DatasetType.completion.value


@json_schema_type
class InstructDataset(BaseDataset):
    type: Literal[DatasetType.instruct.value] = DatasetType.instruct.value


@json_schema_type
class ChatDataset(BaseDataset):
    type: Literal[DatasetType.chat.value] = DatasetType.chat.value


@json_schema_type
class PreferenceDataset(BaseDataset):
    type: Literal[DatasetType.preference.value] = DatasetType.preference.value


@json_schema_type
class EvalDataset(BaseDataset):
    """Evaluation dataset schemas are associated with tasks and not specific training phases"""
    type: Literal[DatasetType.evaluation.value] = DatasetType.evaluation.value


@json_schema_type
class EvalGenerationsDataset(BaseDataset):
    """An evaluation dataset that has been run through a model and has generations attached to it"""
    type: Literal[DatasetType.eval_generations.value] = DatasetType.eval_generations.value

    eval_dataset_id: str


TrainDataset = Annotated[
    Union[
        CompletionDataset,
        InstructDataset,
        ChatDataset,
        PreferenceDataset,
    ],
    Field(discriminator="type"),
]

@json_schema_type
class MixtureTrainDataset(BaseModel):
    # has to be a List of the same type, validated at runtime
    datasets: List[TrainDataset]
    weights: Optional[List[float]]
