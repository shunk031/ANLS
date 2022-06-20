import json
import logging
import os
import pathlib
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from anls.evaluation.schema import EvalResult

logger = logging.getLogger(__name__)


def load_json(file_path: os.PathLike) -> Dict[str, Any]:
    logger.info(f"Load json from {file_path}")

    with open(file_path, "r") as rf:
        json_dict = json.load(rf)
    return json_dict


def save_json(file_path: pathlib.Path, data: EvalResult) -> None:
    assert is_dataclass(data), f"{data} is not a dataclass"
    data_dict = asdict(data)

    with open(file_path, "w") as wf:
        json.dump(data_dict, wf)

    logger.info(
        "All results including per-sample result "
        f"has been correctly saved to {file_path}"
    )
