import pathlib
from typing import Dict, List, Optional

from anls.evaluation.const import ANSWER_TYPES, EVIDENCE_TYPES, REASONING_REQUIREMENTS
from anls.evaluation.schema import (
    EvalResult,
    GoldLabelJson,
    MethodMetrics,
    PerSampleMetric,
    ScoresByTypes,
    SubmissionJson,
)
from anls.metrics import anls_score


def evaluate_json_from_files(
    gold_label_file_path: pathlib.Path,
    submission_file_path: pathlib.Path,
    show_scores_per_answer_type: bool,
    anls_threshold: float,
    question_ids_to_exclude: Optional[List[int]] = None,
) -> EvalResult:

    from anls.common.util import load_json

    gold_label_json = load_json(gold_label_file_path)
    submission_json = load_json(submission_file_path)

    return evaluate_json(
        gold_label_json=gold_label_json,  # type: ignore
        submission_json=submission_json,  # type: ignore
        show_scores_per_answer_type=show_scores_per_answer_type,
        anls_threshold=anls_threshold,
        question_ids_to_exclude=question_ids_to_exclude,
    )


def evaluate_json(
    gold_label_json: GoldLabelJson,
    submission_json: SubmissionJson,
    show_scores_per_answer_type: bool,
    anls_threshold: float,
    question_ids_to_exclude: Optional[List[int]] = None,
) -> EvalResult:

    question_ids_to_exclude = question_ids_to_exclude or []

    res_id_to_index = {int(r["questionId"]): ix for ix, r in enumerate(submission_json)}

    per_sample_metrics: Dict[str, PerSampleMetric] = {}

    total_score = 0.0
    row = 0

    if show_scores_per_answer_type:
        answer_type_total_score = {x: 0 for x in ANSWER_TYPES.keys()}
        answer_type_num_questions = {x: 0 for x in ANSWER_TYPES.keys()}

        evidence_type_total_score = {x: 0 for x in EVIDENCE_TYPES.keys()}
        evidence_type_num_questions = {x: 0 for x in EVIDENCE_TYPES.keys()}

        reasoning_type_total_score = {x: 0 for x in REASONING_REQUIREMENTS.keys()}
        reasoning_type_num_questions = {x: 0 for x in REASONING_REQUIREMENTS.keys()}

    for gt_object in gold_label_json["data"]:

        q_id = int(gt_object["questionId"])
        res_ix = res_id_to_index[q_id]
        det_object = submission_json[res_ix]

        if q_id in question_ids_to_exclude:
            question_result = 0.0
            info = "Question EXCLUDED from the result"

        else:
            info = ""
            # values = []
            # for answer in gt_object["answers"]:
            #     # preprocess both the answers - gt and prediction
            #     gt_answer = " ".join(answer.strip().lower().split())
            #     det_answer = " ".join(det_object["answer"].strip().lower().split())

            #     # dist = levenshtein_distance(answer.lower(), det_object['answer'].lower())
            #     dist = levenshtein_distance(gt_answer, det_answer)
            #     length = max(len(answer.upper()), len(det_object["answer"].upper()))
            #     values.append(0.0 if length == 0 else float(dist) / float(length))

            # question_result = 1.0 - min(values)

            # # if question_result < evaluationParams.anls_threshold:
            # if question_result < anls_threshold:
            #     question_result = 0
            question_result = anls_score(
                prediction=det_object["answer"],
                gold_labels=gt_object["answers"],
                threshold=anls_threshold,
            )

            total_score += question_result

            if show_scores_per_answer_type:
                for q_type in gt_object["answer_type"]:
                    answer_type_total_score[q_type] += question_result
                    answer_type_num_questions[q_type] += 1

                for q_type in gt_object["evidence"]:
                    evidence_type_total_score[q_type] += question_result
                    evidence_type_num_questions[q_type] += 1

                for q_type in gt_object["operation/reasoning"]:
                    reasoning_type_total_score[q_type] += question_result
                    reasoning_type_num_questions[q_type] += 1

        per_sample_metrics[str(gt_object["questionId"])] = PerSampleMetric(
            score=question_result,
            question=gt_object["question"],
            gt=gt_object["answers"],
            det=det_object["answer"],
            info=info,
        )
        row = row + 1

    method_metrics = MethodMetrics(
        score=0
        if len(gold_label_json["data"]) == 0
        else total_score / (len(gold_label_json["data"]) - len(question_ids_to_exclude))
    )

    answer_types_scores = {}
    evidence_types_scores = {}
    operation_types_scores = {}

    if show_scores_per_answer_type:
        for a_type, ref in ANSWER_TYPES.items():
            answer_types_scores[ref] = (
                0.0
                if len(gold_label_json["data"]) == 0
                else answer_type_total_score[a_type]
                / (answer_type_num_questions[a_type])
            )

        for e_type, ref in EVIDENCE_TYPES.items():
            evidence_types_scores[ref] = (
                0.0
                if len(gold_label_json["data"]) == 0
                else evidence_type_total_score[e_type]
                / (evidence_type_num_questions[e_type])
            )

        for r_type, ref in REASONING_REQUIREMENTS.items():
            operation_types_scores[ref] = (
                0.0
                if len(gold_label_json["data"]) == 0
                else reasoning_type_total_score[r_type]
                / (reasoning_type_num_questions[r_type])
            )

    res = EvalResult(
        result=method_metrics,
        scores_by_types=ScoresByTypes(
            answer_types=answer_types_scores,
            evidence_types=evidence_types_scores,
            operation_types=operation_types_scores,
        ),
        per_sample_result=per_sample_metrics,
    )

    return res
