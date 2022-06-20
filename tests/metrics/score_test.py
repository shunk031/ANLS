import pytest
from anls import anls_score
from anls.metrics.dist import normalized_levenshtein_distance as NL


@pytest.mark.parametrize(
    "net_output, expected_ai1, expected_ai2",
    (
        ("The Coca", 0.44, 0.29),
        ("CocaCola", 0.89, 0.47),
        ("Coca cola", 1.00, 0.53),
        ("Cola", 0.44, 0.23),
        ("Cat", 0.22, 0.12),
    ),
)
def test_sim(net_output: str, expected_ai1: float, expected_ai2: float) -> None:

    ai1 = "Coca Cola"
    ai2 = "Coca Cola Company"

    # the score is not case sentive, but space sensitive
    ai1 = " ".join(ai1.strip().lower().split())
    ai2 = " ".join(ai2.strip().lower().split())
    net_output = " ".join(net_output.strip().lower().split())

    assert 1 - NL(net_output, ai1) == pytest.approx(expected_ai1, 0.1)
    assert 1 - NL(net_output, ai2) == pytest.approx(expected_ai2, 0.1)


@pytest.mark.parametrize(
    "net_output, expected_score",
    (
        ("The Coca", 0.00),
        ("CocaCola", 0.89),
        ("Coca cola", 1.00),
        ("Cola", 0.00),
        ("Cat", 0.00),
    ),
)
def test_anls_score(net_output: str, expected_score: float) -> None:

    ai1 = "Coca Cola"
    ai2 = "Coca Cola Company"

    score = anls_score(prediction=net_output, gold_labels=[ai1, ai2], threshold=0.5)
    assert score == pytest.approx(expected_score, 0.1)
