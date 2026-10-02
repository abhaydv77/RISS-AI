import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import score_pairs


def make_pairs():
    return [
        {
            "query": "Brand: FitNova. Industry: Fitness.",
            "passage": "Creator: Priya Sharma. Primary niche: fitness.",
            "brand_id": "b01",
            "creator_id": "c01",
        },
        {
            "query": "Brand: FitNova. Industry: Fitness.",
            "passage": "Creator: Kai Tanaka. Primary niche: gaming.",
            "brand_id": "b01",
            "creator_id": "c08",
        },
        {
            "query": "Brand: GlowEssence. Industry: Beauty.",
            "passage": "Creator: Sofia Chen. Primary niche: beauty.",
            "brand_id": "b02",
            "creator_id": "c05",
        },
    ]


def test_score_pairs_returns_correct_count():
    model = MagicMock()
    model.predict.return_value = [0.5, 0.3, 0.9]
    pairs = make_pairs()
    results = score_pairs(model, pairs)
    assert len(results) == 3


def test_score_pairs_preserves_order():
    model = MagicMock()
    model.predict.return_value = [0.1, 0.2, 0.3]
    pairs = make_pairs()
    results = score_pairs(model, pairs)
    assert results[0]["creator_id"] == "c01"
    assert results[1]["creator_id"] == "c08"
    assert results[2]["creator_id"] == "c05"


def test_score_pairs_preserves_brand_id():
    model = MagicMock()
    model.predict.return_value = [0.1, 0.2, 0.3]
    pairs = make_pairs()
    results = score_pairs(model, pairs)
    assert results[0]["brand_id"] == "b01"
    assert results[1]["brand_id"] == "b01"
    assert results[2]["brand_id"] == "b02"


def test_score_pairs_returns_float_scores():
    model = MagicMock()
    model.predict.return_value = [0.5, 0.3, 0.9]
    pairs = make_pairs()
    results = score_pairs(model, pairs)
    for r in results:
        assert isinstance(r["score"], float)


def test_score_pairs_empty_input():
    model = MagicMock()
    results = score_pairs(model, [])
    assert results == []
    model.predict.assert_not_called()


def test_score_pairs_calls_predict_with_query_passage():
    model = MagicMock()
    model.predict.return_value = [0.5, 0.3, 0.9]
    pairs = make_pairs()
    score_pairs(model, pairs)
    call_args = model.predict.call_args[0][0]
    assert call_args == [
        ("Brand: FitNova. Industry: Fitness.", "Creator: Priya Sharma. Primary niche: fitness."),
        ("Brand: FitNova. Industry: Fitness.", "Creator: Kai Tanaka. Primary niche: gaming."),
        ("Brand: GlowEssence. Industry: Beauty.", "Creator: Sofia Chen. Primary niche: beauty."),
    ]


if __name__ == "__main__":
    test_score_pairs_returns_correct_count()
    test_score_pairs_preserves_order()
    test_score_pairs_preserves_brand_id()
    test_score_pairs_returns_float_scores()
    test_score_pairs_empty_input()
    test_score_pairs_calls_predict_with_query_passage()
    print("All tests passed.")
