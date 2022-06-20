import pathlib

import pytest
from anls.common.exception import AnlsException
from anls.evaluation.validate_data import (
    _check_data_key_exists,
    _check_dataset_name_key_exists,
    _check_json_format,
    _check_question_and_answers,
    _check_two_json_length,
    _check_two_questions_length,
    validate_data,
    validate_data_from_files,
)


def test_check_data_key_exists():
    with pytest.raises(AnlsException):
        _check_data_key_exists({"foo": "bar"})

    _check_data_key_exists({"data": "baz"})


def test_check_dataset_name_key_exists():
    with pytest.raises(AnlsException):
        _check_dataset_name_key_exists({"foo": "bar"})

    _check_dataset_name_key_exists({"dataset_name": "baz"})


def test_check_json_format():

    bad_submission_json = {"data": "foo", "dataset_name": "bar"}
    with pytest.raises(AnlsException):
        _check_json_format(bad_submission_json)

    good_submission_json = [
        {"questionId": 1, "answer": "foo"},
        {"questionId": 2, "answer": "bar"},
    ]
    _check_json_format(good_submission_json)


def test_check_two_json_length():

    submission_json = [
        {"questionId": 1, "answer": "foo"},
        {"questionId": 2, "answer": "bar"},
    ]
    gold_label_json = {
        "dataset_name": "foo",
        "dataset_version": "1.0",
        "dataset_split": "train",
        "data": [
            {
                "questionId": 1,
                "question": "foo",
                "image_local_name": "bar",
                "image_url": "https://example.com",
                "ocr_output_file": "path_to_ocr_file",
                "answers": ["baz"],
                "data_split": "train",
            },
            {
                "questionId": 2,
                "question": "foo",
                "image_local_name": "bar",
                "image_url": "https://example.com",
                "ocr_output_file": "path_to_ocr_file",
                "answers": ["baz"],
                "data_split": "train",
            },
        ],
    }

    _check_two_json_length(
        gold_label_json=gold_label_json,
        submission_json=submission_json,
    )

    submission_json.append("unexpected data")
    with pytest.raises(AnlsException):
        _check_two_json_length(
            gold_label_json=gold_label_json,
            submission_json=submission_json,
        )


def _check_two_question_length():
    submission_json = [
        {"questionId": 1, "answer": "foo"},
        {"questionId": 2, "answer": "bar"},
    ]
    gold_label_json = {
        "dataset_name": "foo",
        "dataset_version": "1.0",
        "dataset_split": "train",
        "data": [
            {
                "questionId": 1,
                "question": "foo",
                "image_local_name": "bar",
                "image_url": "https://example.com",
                "ocr_output_file": "path_to_ocr_file",
                "answers": ["baz"],
                "data_split": "train",
            },
            {
                "questionId": 2,
                "question": "foo",
                "image_local_name": "bar",
                "image_url": "https://example.com",
                "ocr_output_file": "path_to_ocr_file",
                "answers": ["baz"],
                "data_split": "train",
            },
        ],
    }
    _check_two_questions_length(
        gold_label_json=gold_label_json,
        submission_json=submission_json,
    )

    submission_json.append({"questionId": 3, "answer": "unexpected answer"})
    with pytest.raises(AnlsException):
        _check_two_questions_length(
            gold_label_json=gold_label_json,
            submission_json=submission_json,
        )


def test_check_question_and_answers():
    submission_json = [
        {"questionId": 1, "answer": "foo"},
        {"questionId": 2, "answer": "bar"},
    ]
    gold_label_json = {
        "dataset_name": "foo",
        "dataset_version": "1.0",
        "dataset_split": "train",
        "data": [
            {
                "questionId": 1,
                "question": "foo",
                "image_local_name": "bar",
                "image_url": "https://example.com",
                "ocr_output_file": "path_to_ocr_file",
                "answers": ["baz"],
                "data_split": "train",
            },
            {
                "questionId": 2,
                "question": "foo",
                "image_local_name": "bar",
                "image_url": "https://example.com",
                "ocr_output_file": "path_to_ocr_file",
                "answers": ["baz"],
                "data_split": "train",
            },
        ],
    }

    res_id_to_idx = {int(r["questionId"]): ix for ix, r in enumerate(submission_json)}
    for gold_label_data in gold_label_json["data"]:
        _check_question_and_answers(
            gold_label_data=gold_label_data,
            submission_json=submission_json,
            res_id_to_idx=res_id_to_idx,
        )


def test_validate_data():
    submission_json = [
        {"questionId": 1, "answer": "foo"},
        {"questionId": 2, "answer": "bar"},
    ]
    gold_label_json = {
        "dataset_name": "foo",
        "dataset_version": "1.0",
        "dataset_split": "train",
        "data": [
            {
                "questionId": 1,
                "question": "foo",
                "image_local_name": "bar",
                "image_url": "https://example.com",
                "ocr_output_file": "path_to_ocr_file",
                "answers": ["baz"],
                "data_split": "train",
            },
            {
                "questionId": 2,
                "question": "foo",
                "image_local_name": "bar",
                "image_url": "https://example.com",
                "ocr_output_file": "path_to_ocr_file",
                "answers": ["baz"],
                "data_split": "train",
            },
        ],
    }
    validate_data(
        gold_label_json=gold_label_json,
        submission_json=submission_json,
    )


def test_validate_data_from_files():
    root_dir = pathlib.Path(__file__).resolve().parents[2]
    test_fixtures_dir = root_dir / "test_fixtures"
    json_dir = test_fixtures_dir / "evaluation" / "validate_data"

    validate_data_from_files(
        gold_label_file_path=json_dir / "gold_label.json",
        submission_file_path=json_dir / "submission.json",
    )
