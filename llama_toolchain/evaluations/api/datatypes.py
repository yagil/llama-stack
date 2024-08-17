# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Union
from pydantic import BaseModel


class EvaluationJob(BaseModel):
    job_uuid: str


class EvaluationJobLogStream(BaseModel):
    job_uuid: str


# Model Evals needs
# - model to be evaluated
# - dataset
# - computation info  # batch inference or not , batch size , etc
# what are the most common ways Datasets are mentioned in evals ?
class TextGenerationMetric(Enum):
    perplexity = "perplexity"
    rouge = "rouge"
    bleu = "bleu"


class QuestionAnsweringMetric(Enum):
    em = "em"
    f1 = "f1"


class SummarizationMetric(Enum):
    rouge = "rouge"
    bleu = "bleu"


Metric = Union[TextGenerationMetric, QuestionAnsweringMetric, SummarizationMetric]


@json_schema_type
class PromptTemplate(BaseModel):
    """How to format the prompt for the model"""

    template: str


@json_schema_type
class ModelCandidate(BaseModel):
    """What are we evaluating ? Usually its a combination of model and prompt"""

    model: str
    sampling_params: SamplingParams
    prompt_format: PromptTemplate


@json_schema_type
class AgentCandidate(BaseModel):
    """Evaluate an agent"""

    agent: AgenticSystemInstanceConfig
    prompt_format: PromptTemplate


# == endpoints ==
# /evals/tasks -- list all supported tasks
# /evals/create -- create a new eval job
# /evals/cancel -- cancel an eval job
# /evals/status -- get status of an eval job
# /evals/results -- get results of an eval job
# === Requirements ===
# - Need a quick interactive way to run evals for e2e tests / debugging
# - Keep it simple to translate prototype to prod pipeline
# What are the most common ways Datasets are mentioned in evals ?
# fair-evals -- all datasets are jsonl files in a directory
# In HF, how do they do it ?
# Why do we not use EluetherAI's lm-eval-harness internally ?
# Steps of Eval
# - data loading
# - jsonl format
# - or hive table
# - generation
# - offline inference / remote api
# -
# - comparison
# model based eval tasks
# Use another fixed judge to evaluate the generations
# Model outputs log-probs
# Hww long do eval tasks take ?
# - some have 100s of data items
# - mmlu has lot of subjects ~ 16k
# - Should be done in a few mins
# Need composbale apis -- when user already has generations


class Task:
    name: str
    description: str
    dataset_id: str
    metrics: List[Metric]


# Internal representation of a task (which is a group of tasks)
# and does some aggregation of the results of the tasks
class TaskGroup:
    tasks: List[Task]
    metrics: List[Metric]
