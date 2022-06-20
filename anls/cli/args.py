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
    return parser.parse_args()
