from dataclasses import dataclass
from typing import Dict, List

try:
    from typing import TypedDict  # type: ignore
except ImportError:
    from typing_extensions import TypedDict


@dataclass
class MethodMetrics(object):
    score: float


@dataclass
class ScoresByTypes(object):
    answer_types: Dict[str, float]
    evidence_types: Dict[str, float]
    operation_types: Dict[str, float]


@dataclass
class PerSampleMetric(object):
    score: float
    question: str
    gt: List[str]
    det: str
    info: str


@dataclass
class EvalResult(object):
    result: MethodMetrics
    scores_by_types: ScoresByTypes
    per_sample_result: Dict[str, PerSampleMetric]


class GoldLabelData(TypedDict):
    questionId: int
    question: str
    image_local_name: str
    image_url: str
    ocr_output_file: str
    answers: List[str]
    data_split: str


class GoldLabelJson(TypedDict):
    dataset_name: str
    dataset_version: str
    dataset_split: str
    data: List[GoldLabelData]


class SubmissionDict(TypedDict):
    questionId: int
    answer: str


SubmissionJson = List[SubmissionDict]
