import logging

from anls.evaluation.const import ANSWER_TYPES, EVIDENCE_TYPES, REASONING_REQUIREMENTS
from anls.evaluation.schema import EvalResult

logger = logging.getLogger(__name__)


def display_results(results: EvalResult, show_answer_types: bool) -> None:

    logger.info("\nOverall ANLS: {:2.4f}".format(results.result.score))

    if show_answer_types:
        logger.info("\nAnswer types:")
        for a_type in ANSWER_TYPES.values():
            logger.info(
                f"\t{a_type:12s} {results.scores_by_types.answer_types[a_type]:2.4f}"
            )

        logger.info("\nEvidence types:")
        for e_type in EVIDENCE_TYPES.values():
            logger.info(
                f"\t{e_type:12s} {results.scores_by_types.evidence_types[e_type]:2.4f}"
            )

        logger.info("\nOperation required:")
        for r_type in REASONING_REQUIREMENTS.values():
            logger.info(
                f"\t{r_type:12s} {results.scores_by_types.operation_types[r_type]:2.4f}"
            )
