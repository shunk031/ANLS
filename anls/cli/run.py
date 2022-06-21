import logging
import os
import pathlib
from typing import Optional

from anls.cli.args import parse_args
from anls.common.util import save_json
from anls.evaluation import display_results, evaluate_json, validate_data

if os.environ.get("ANLS_DEBUG"):
    LEVEL = logging.DEBUG
else:
    level_name = os.environ.get("ANLS_LOG_LEVEL", "INFO")
    LEVEL = logging._nameToLevel.get(level_name, logging.INFO)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL
)


def evaluate_json_from_args(
    gold_label_file: pathlib.Path,
    submission_file: pathlib.Path,
    answer_types: bool,
    output_dir: Optional[pathlib.Path],
    anls_threshold: float,
    results_json: str = "results.json",
) -> None:

    if answer_types:
        raise NotImplementedError(
            "The Ground truth file requires ANLS breakdown information by type, "
            "but unfortunately such information is not publicly available as of March 25th."
        )

    from anls.common.util import load_gold_label_json, load_submission_json

    gold_label_json = load_gold_label_json(gold_label_file)
    submission_json = load_submission_json(submission_file)

    validate_data(
        gold_label_json=gold_label_json,
        submission_json=submission_json,
    )

    # Evaluate method
    results = evaluate_json(
        gold_label_json=gold_label_json,
        submission_json=submission_json,
        show_scores_per_answer_type=answer_types,
        anls_threshold=anls_threshold,
    )

    display_results(results, answer_types)

    if output_dir:
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        results_output_name = output_dir / results_json
        save_json(results_output_name, results)


def run(prog: Optional[str] = None) -> None:
    args = parse_args(prog)

    evaluate_json_from_args(
        gold_label_file=args.gold_label_file,
        submission_file=args.submission_file,
        answer_types=args.answer_types,
        output_dir=args.output_dir,
        anls_threshold=args.anls_threshold,
    )
