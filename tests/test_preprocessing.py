import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (
    brand_to_query,
    creator_to_passage,
    load_brands,
    load_creators,
    make_pairs,
)


def test_load_brands():
    brands = load_brands()
    assert isinstance(brands, list)
    assert len(brands) > 0
    assert "brand_id" in brands[0]
    assert "brand_name" in brands[0]


def test_load_creators():
    creators = load_creators()
    assert isinstance(creators, list)
    assert len(creators) > 0
    assert "creator_id" in creators[0]
    assert "name" in creators[0]


def test_brand_to_query_returns_string():
    brands = load_brands()
    query = brand_to_query(brands[0])
    assert isinstance(query, str)
    assert len(query) > 0


def test_brand_to_query_deterministic():
    brands = load_brands()
    q1 = brand_to_query(brands[0])
    q2 = brand_to_query(brands[0])
    assert q1 == q2


def test_brand_to_query_contains_key_fields():
    brands = load_brands()
    brand = brands[0]
    query = brand_to_query(brand)
    assert brand["brand_name"] in query
    assert brand["industry"] in query
    assert brand["campaign_goal"] in query


def test_brand_to_query_handles_missing_fields():
    minimal = {"brand_id": "test", "brand_name": "TestBrand"}
    query = brand_to_query(minimal)
    assert isinstance(query, str)
    assert "TestBrand" in query


def test_creator_to_passage_returns_string():
    creators = load_creators()
    passage = creator_to_passage(creators[0])
    assert isinstance(passage, str)
    assert len(passage) > 0


def test_creator_to_passage_deterministic():
    creators = load_creators()
    p1 = creator_to_passage(creators[0])
    p2 = creator_to_passage(creators[0])
    assert p1 == p2


def test_creator_to_passage_contains_key_fields():
    creators = load_creators()
    creator = creators[0]
    passage = creator_to_passage(creator)
    assert creator["name"] in passage
    assert creator["primary_niche"] in passage
    assert creator["bio"] in passage


def test_creator_to_passage_handles_missing_fields():
    minimal = {"creator_id": "test", "name": "TestCreator"}
    passage = creator_to_passage(minimal)
    assert isinstance(passage, str)
    assert "TestCreator" in passage


def test_make_pairs_count():
    brands = load_brands()
    creators = load_creators()
    pairs = make_pairs(brands[0], creators)
    assert len(pairs) == len(creators)


def test_make_pairs_structure():
    brands = load_brands()
    creators = load_creators()
    pairs = make_pairs(brands[0], creators)
    pair = pairs[0]
    assert "query" in pair
    assert "passage" in pair
    assert "brand_id" in pair
    assert "creator_id" in pair


def test_make_pairs_id_preservation():
    brands = load_brands()
    creators = load_creators()
    pairs = make_pairs(brands[0], creators)
    for i, pair in enumerate(pairs):
        assert pair["brand_id"] == brands[0]["brand_id"]
        assert pair["creator_id"] == creators[i]["creator_id"]


def test_make_pairs_query_passage_content():
    brands = load_brands()
    creators = load_creators()
    pairs = make_pairs(brands[0], creators)
    pair = pairs[0]
    assert pair["query"] == brand_to_query(brands[0])
    assert pair["passage"] == creator_to_passage(creators[0])


if __name__ == "__main__":
    test_load_brands()
    test_load_creators()
    test_brand_to_query_returns_string()
    test_brand_to_query_deterministic()
    test_brand_to_query_contains_key_fields()
    test_brand_to_query_handles_missing_fields()
    test_creator_to_passage_returns_string()
    test_creator_to_passage_deterministic()
    test_creator_to_passage_contains_key_fields()
    test_creator_to_passage_handles_missing_fields()
    test_make_pairs_count()
    test_make_pairs_structure()
    test_make_pairs_id_preservation()
    test_make_pairs_query_passage_content()
    print("All tests passed.")
