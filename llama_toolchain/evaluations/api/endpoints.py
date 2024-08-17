# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Protocol

from llama_models.schema_utils import webmethod

from pydantic import BaseModel

from llama_models.llama3_1.api.datatypes import *  # noqa: F403
from .datatypes import *  # noqa: F403
from llama_toolchain.dataset.api.datatypes import *  # noqa: F403
from llama_toolchain.common.training_types import *  # noqa: F403



@json_schema_type
class EvalCreateRequest(BaseModel):
    candidate: ModelCandidate | AgentCandidate
    tasks: List[Task]


# where do the generations go?
@json_schema_type
class EvalGenerationsRequest(BaseModel):
    candidate: ModelCandidate | AgentCandidate
    tasks: List[Task]

@json_schema_type
class EvalScoringRequest(BaseModel):
    candidate: ModelCandidate | AgentCandidate
    tasks: List[Task]

@json_schema_type
class EvalCreateResponse(BaseModel):
    job_id: str


@json_schema_type
class EvaluationJobArtifactsResponse(BaseModel):
    """Artifacts of a evaluation job."""

    job_uuid: str


class Evaluations(Protocol):
    @webmethod(route="/evals/supported_tasks", method="GET")
    def supported_tasks(self) -> List[str]: ...

    @webmethod(route="/evals/create")
    def create_evaluation_job(
        self,
        request: EvalCreateRequest,
    ) -> EvalCreateResponse: ...

    @webmethod(route="/evaluate/jobs")
    def get_evaluation_jobs(self) -> List[EvaluationJob]: ...

    @webmethod(route="/evals/job/status", method="GET")
    def get_evaluation_job_status(
        self, job_uuid: str
    ) -> EvaluationJobStatusResponse: ...

    # sends SSE stream of logs
    @webmethod(route="/evaluate/job/logs")
    def get_evaluation_job_logstream(self, job_uuid: str) -> EvaluationJobLogStream: ...

    @webmethod(route="/evals/job/cancel")
    def cancel_evaluation_job(self, job_uuid: str) -> None: ...

    @webmethod(route="/evaluate/job/artifacts")
    def get_evaluation_job_artifacts(
        self, job_uuid: str
    ) -> EvaluationJobArtifactsResponse: ...
