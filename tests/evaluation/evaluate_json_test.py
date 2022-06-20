import pathlib

import pytest
from anls.common.util import load_json
from anls.evaluation.evaluate_json import evaluate_json


def test_evaluate_json():
    root_dir = pathlib.Path(__file__).resolve().parents[2]
    test_fixtures_dir = root_dir / "test_fixtures"
    json_dir = test_fixtures_dir / "evaluation" / "evaluate_json"

    gold_label_json = load_json(json_dir / "gold_label.json")
    submission_json = load_json(json_dir / "submission.json")

    results = evaluate_json(
        gold_label_json=gold_label_json,
        submission_json=submission_json,
        show_scores_per_answer_type=False,
        anls_threshold=0.5,
    )

    assert results.result.score == pytest.approx(0.378, 0.001)

    expect_per_sample_results = {
        "1": {
            "score": 0.0,
        },
        "2": {
            "score": 0.89,
        },
        "3": {
            "score": 1.0,
        },
        "4": {
            "score": 0.0,
        },
        "5": {
            "score": 0.0,
        },
    }

    actual_per_sample_results = results.per_sample_result
    for k in actual_per_sample_results.keys():
        expect = expect_per_sample_results[k]["score"]
        actual = actual_per_sample_results[k].score
        assert actual == pytest.approx(expect, 0.1)
