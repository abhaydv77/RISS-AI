import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import compute_metrics


def test_perfect_ranking():
    labels = {"c01": "good", "c02": "good", "c03": "poor", "c04": "maybe"}
    ranked = ["c01", "c02", "c04", "c03"]
    metrics = compute_metrics(ranked, labels)
    assert metrics["precision_at_3"] == 2 / 3
    assert metrics["precision_at_5"] == 2 / 4
    assert metrics["recall_at_3"] == 1.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg_at_5"] == 1.0


def test_worst_ranking():
    labels = {"c01": "good", "c02": "good", "c03": "poor", "c04": "poor"}
    ranked = ["c03", "c04", "c01", "c02"]
    metrics = compute_metrics(ranked, labels)
    assert metrics["precision_at_3"] == 1 / 3
    assert metrics["recall_at_3"] == 1 / 2
    assert metrics["mrr"] == 1 / 3


def test_mrr_first_position():
    labels = {"c01": "good", "c02": "poor"}
    ranked = ["c01", "c02"]
    metrics = compute_metrics(ranked, labels)
    assert metrics["mrr"] == 1.0


def test_mrr_second_position():
    labels = {"c01": "good", "c02": "poor"}
    ranked = ["c02", "c01"]
    metrics = compute_metrics(ranked, labels)
    assert metrics["mrr"] == 0.5


def test_mrr_no_relevant():
    labels = {"c01": "poor", "c02": "poor"}
    ranked = ["c01", "c02"]
    metrics = compute_metrics(ranked, labels)
    assert metrics["mrr"] == 0.0


def test_recall_at_k_partial():
    labels = {"c01": "good", "c02": "good", "c03": "good", "c04": "poor"}
    ranked = ["c01", "c04", "c02", "c03"]
    metrics = compute_metrics(ranked, labels)
    assert metrics["recall_at_3"] == 2 / 3
    assert metrics["recall_at_5"] == 1.0


def test_precision_at_k():
    labels = {"c01": "good", "c02": "poor", "c03": "good", "c04": "poor"}
    ranked = ["c01", "c02", "c03", "c04"]
    metrics = compute_metrics(ranked, labels)
    assert metrics["precision_at_3"] == 2 / 3
    assert metrics["precision_at_5"] == 2 / 4


def test_ndcg_perfect():
    labels = {"c01": "good", "c02": "maybe", "c03": "poor"}
    ranked = ["c01", "c02", "c03"]
    metrics = compute_metrics(ranked, labels)
    assert metrics["ndcg_at_5"] == 1.0


def test_ndcg_imperfect():
    labels = {"c01": "good", "c02": "maybe", "c03": "poor"}
    ranked = ["c03", "c02", "c01"]
    metrics = compute_metrics(ranked, labels)
    assert metrics["ndcg_at_5"] < 1.0
    assert metrics["ndcg_at_5"] > 0.0


def test_empty_labels():
    ranked = ["c01", "c02"]
    metrics = compute_metrics(ranked, {})
    assert metrics["precision_at_3"] == 0.0
    assert metrics["recall_at_3"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["ndcg_at_5"] == 0.0


def test_maybe_not_relevant_for_precision():
    labels = {"c01": "maybe", "c02": "good", "c03": "poor"}
    ranked = ["c01", "c02", "c03"]
    metrics = compute_metrics(ranked, labels)
    assert metrics["precision_at_3"] == 1 / 3
    assert metrics["recall_at_3"] == 1.0


if __name__ == "__main__":
    test_perfect_ranking()
    test_worst_ranking()
    test_mrr_first_position()
    test_mrr_second_position()
    test_mrr_no_relevant()
    test_recall_at_k_partial()
    test_precision_at_k()
    test_ndcg_perfect()
    test_ndcg_imperfect()
    test_empty_labels()
    test_maybe_not_relevant_for_precision()
    print("All tests passed.")
