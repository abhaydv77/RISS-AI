import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ranker import rank_brand


def make_creators(n=5):
    return [
        {
            "creator_id": f"c{i:02d}",
            "name": f"Creator {i}",
            "primary_niche": "fitness",
            "bio": "Test bio",
            "followers": 10000 * (i + 1),
        }
        for i in range(1, n + 1)
    ]


def make_brand():
    return {
        "brand_id": "b01",
        "brand_name": "FitNova",
        "industry": "Fitness & Wellness",
        "product": "Home workout equipment",
        "campaign_goal": "Drive awareness",
        "target_audience": "Young women",
        "target_age_range": "18-30",
        "target_gender": "female",
        "target_locations": ["India"],
        "required_creator_niches": ["fitness"],
        "preferred_creator_niches": ["fitness", "wellness"],
        "creator_size_preference": "mid-tier",
        "minimum_followers": 50000,
        "maximum_followers": 500000,
        "content_types": ["workout"],
        "platforms": ["instagram"],
        "tone": "energetic",
        "mandatory_requirements": ["must be based in India"],
        "preferred_traits": ["authentic"],
        "excluded_traits": ["luxury-focused"],
    }


def test_rank_brand_returns_k_results():
    model = MagicMock()
    model.predict.return_value = [0.5, 0.3, 0.9, 0.1, 0.7]
    brand = make_brand()
    creators = make_creators(5)
    results = rank_brand(model, brand, creators, k=3)
    assert len(results) == 3


def test_rank_brand_ordering():
    model = MagicMock()
    model.predict.return_value = [0.5, 0.3, 0.9, 0.1, 0.7]
    brand = make_brand()
    creators = make_creators(5)
    results = rank_brand(model, brand, creators, k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rank_brand_top_k():
    model = MagicMock()
    model.predict.return_value = [0.5, 0.3, 0.9, 0.1, 0.7]
    brand = make_brand()
    creators = make_creators(5)
    results = rank_brand(model, brand, creators, k=2)
    assert len(results) == 2
    assert results[0]["score"] == 0.9
    assert results[1]["score"] == 0.7


def test_rank_brand_k_larger_than_creators():
    model = MagicMock()
    model.predict.return_value = [0.5, 0.3]
    brand = make_brand()
    creators = make_creators(2)
    results = rank_brand(model, brand, creators, k=10)
    assert len(results) == 2


def test_rank_brand_invalid_k():
    model = MagicMock()
    brand = make_brand()
    creators = make_creators(3)
    try:
        rank_brand(model, brand, creators, k=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass
    try:
        rank_brand(model, brand, creators, k=-1)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_rank_brand_id_preservation():
    model = MagicMock()
    model.predict.return_value = [0.5, 0.3, 0.9, 0.1, 0.7]
    brand = make_brand()
    creators = make_creators(5)
    results = rank_brand(model, brand, creators, k=5)
    result_creator_ids = {r["creator_id"] for r in results}
    assert result_creator_ids == {"c01", "c02", "c03", "c04", "c05"}
    for r in results:
        assert r["brand_id"] == "b01"


def test_rank_brand_score_ordering():
    model = MagicMock()
    model.predict.return_value = [0.2, 0.8, 0.4, 0.6, 0.1]
    brand = make_brand()
    creators = make_creators(5)
    results = rank_brand(model, brand, creators, k=5)
    scores = [r["score"] for r in results]
    assert scores == [0.8, 0.6, 0.4, 0.2, 0.1]


if __name__ == "__main__":
    test_rank_brand_returns_k_results()
    test_rank_brand_ordering()
    test_rank_brand_top_k()
    test_rank_brand_k_larger_than_creators()
    test_rank_brand_invalid_k()
    test_rank_brand_id_preservation()
    test_rank_brand_score_ordering()
    print("All tests passed.")
