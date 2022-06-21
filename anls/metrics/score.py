from typing import List

from anls.metrics.dist import normalized_levenshtein_distance as NL


def similatiry_function(prediction: str, gold_label: str, threshold: float) -> float:
    nl_score = NL(prediction, gold_label)
    return 1 - nl_score if nl_score < threshold else 0.0


def anls_score(prediction: str, gold_labels: List[str], threshold: float) -> float:

    # not case sensitive, but space sensitive
    y_pred = " ".join(prediction.strip().lower().split())

    anls_scores: List[float] = []
    for gold_label in gold_labels:

        # not case sensitive, but space sensitive
        y_true = " ".join(gold_label.strip().lower().split())

        anls_score = similatiry_function(y_pred, y_true, threshold)
        anls_scores.append(anls_score)

    score = max(anls_scores)

    return score
