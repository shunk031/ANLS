import pathlib
from typing import Any, Dict

from anls.common.exception import AnlsException
from anls.evaluation.schema import GoldLabelData, GoldLabelJson, SubmissionJson


def _check_data_key_exists(gold_label_json: GoldLabelJson) -> None:
    if "data" not in gold_label_json:
        raise AnlsException("The GT file is not valid (no data key)")


def _check_dataset_name_key_exists(gold_label_json: GoldLabelJson) -> None:
    if "dataset_name" not in gold_label_json:
        raise AnlsException("The GT file is not valid (no dataset_name key)")


def _check_json_format(submission_json: SubmissionJson) -> None:
    if not isinstance(submission_json, list):
        raise AnlsException("The Det file is not valid (root item must be an array)")


def _check_two_json_length(
    gold_label_json: GoldLabelJson, submission_json: SubmissionJson
) -> None:

    len_submission = len(submission_json)
    len_gold_label = len(gold_label_json["data"])

    if len_submission != len_gold_label:
        raise AnlsException(
            "The Det file is not valid (invalid number of answers."
            f"Expected: {len_gold_label} "
            f"Found: {len_submission}"
            ")"
        )


def _check_two_questions_length(
    gold_label_json: GoldLabelJson, submission_json: SubmissionJson
) -> None:

    q_submission = sorted([r["questionId"] for r in submission_json])
    q_gold_label = sorted([r["questionId"] for r in gold_label_json["data"]])

    if not (q_submission == q_gold_label):
        raise AnlsException("The Det file is not valid. Question IDs must much GT")


def _check_question_and_answers(
    gold_label_data: GoldLabelData,
    submission_json: SubmissionJson,
    res_id_to_idx: Dict[int, Any],
) -> None:

    try:
        q_id = int(gold_label_data["questionId"])
        res_idx = res_id_to_idx[q_id]
    except Exception as err:
        raise AnlsException(
            f"The Det file is not valid. Question {q_id} not present"
        ) from err
    else:
        submission = submission_json[res_idx]

        if "answer" not in submission:
            raise AnlsException(
                f"Question {submission['questionId']} not valid (no answer key)"
            )

        if isinstance(submission["answer"], list):
            raise AnlsException(
                f"Question {submission['questionId']} not valid (answer key has to be a single string)"
            )


def validate_data(
    gold_label_json: GoldLabelJson,
    submission_json: SubmissionJson,
) -> None:

    _check_data_key_exists(gold_label_json)
    _check_dataset_name_key_exists(gold_label_json)
    _check_json_format(submission_json)

    _check_two_json_length(gold_label_json, submission_json)

    _check_two_questions_length(gold_label_json, submission_json)

    res_id_to_idx = {int(r["questionId"]): ix for ix, r in enumerate(submission_json)}
    for gold_label in gold_label_json["data"]:
        _check_question_and_answers(gold_label, submission_json, res_id_to_idx)


def validate_data_from_files(
    gold_label_file_path: pathlib.Path,
    submission_file_path: pathlib.Path,
) -> None:
    from anls.common.util import load_gold_label_json, load_submission_json

    gold_label_json = load_gold_label_json(gold_label_file_path)
    submission_json = load_submission_json(submission_file_path)

    validate_data(
        gold_label_json=gold_label_json,
        submission_json=submission_json,
    )
