import argparse
import pathlib
from typing import Optional


def parse_args(prog: Optional[str] = None) -> argparse.Namespace:
    description = "Evaluation command using ANLS"
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "--gold-label-file",
        type=pathlib.Path,
        help="Path of the Ground Truth file.",
        required=True,
    )
    parser.add_argument(
        "--submission-file",
        type=pathlib.Path,
        help="Path of your method's results file.",
        required=True,
    )
    parser.add_argument(
        "--anls-threshold",
        type=float,
        default=0.5,
        help="ANLS threshold to use (See Scene-Text VQA paper for more info.).",
    )
    parser.add_argument(
        "--answer-types",
        action="store_true",
        default=False,
        help="Score break down by answer types (special gt file required).",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Path to a directory where to copy the file 'results.json' that contains per-sample results.",
        required=True,
    )
    return parser.parse_args()
