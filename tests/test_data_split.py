import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_split import (
    LABEL_TO_GRADE,
    build_training_examples,
    create_split_manifest,
    labels_for_split,
    split_brands,
)


def test_label_mapping():
    assert LABEL_TO_GRADE["poor"] == 0
    assert LABEL_TO_GRADE["maybe"] == 1
    assert LABEL_TO_GRADE["good"] == 2


def test_split_brands_no_overlap():
    brand_ids = [f"b{i:02d}" for i in range(1, 11)]
    split = split_brands(brand_ids, seed=42)
    train_set = set(split["train"])
    val_set = set(split["val"])
    test_set = set(split["test"])
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_split_brands_complete_coverage():
    brand_ids = [f"b{i:02d}" for i in range(1, 11)]
    split = split_brands(brand_ids, seed=42)
    all_brands = set(split["train"]) | set(split["val"]) | set(split["test"])
    assert all_brands == set(brand_ids)


def test_split_deterministic():
    brand_ids = [f"b{i:02d}" for i in range(1, 11)]
    split1 = split_brands(brand_ids, seed=42)
    split2 = split_brands(brand_ids, seed=42)
    assert split1 == split2


def test_split_manifest_structure():
    labels = [
        {"brand_id": "b01", "creator_id": "c01", "label": "good"},
        {"brand_id": "b01", "creator_id": "c02", "label": "poor"},
        {"brand_id": "b02", "creator_id": "c03", "label": "maybe"},
        {"brand_id": "b03", "creator_id": "c04", "label": "good"},
    ]
    manifest = create_split_manifest(labels, seed=42)
    assert "seed" in manifest
    assert "split" in manifest
    assert "train" in manifest["split"]
    assert "val" in manifest["split"]
    assert "test" in manifest["split"]


def test_labels_for_split():
    labels = [
        {"brand_id": "b01", "creator_id": "c01", "label": "good"},
        {"brand_id": "b02", "creator_id": "c02", "label": "poor"},
        {"brand_id": "b03", "creator_id": "c03", "label": "maybe"},
    ]
    result = labels_for_split(labels, ["b01", "b03"])
    assert len(result) == 2
    assert all(l["brand_id"] in {"b01", "b03"} for l in result)


def test_build_training_examples():
    labels = [
        {"brand_id": "b01", "creator_id": "c01", "label": "good"},
        {"brand_id": "b01", "creator_id": "c02", "label": "poor"},
    ]
    brands = {
        "b01": {
            "brand_id": "b01",
            "brand_name": "TestBrand",
            "industry": "Fitness",
            "product": "Equipment",
        }
    }
    creators = {
        "c01": {
            "creator_id": "c01",
            "name": "Creator1",
            "primary_niche": "fitness",
            "bio": "Test bio",
        },
        "c02": {
            "creator_id": "c02",
            "name": "Creator2",
            "primary_niche": "gaming",
            "bio": "Test bio 2",
        },
    }
    examples = build_training_examples(labels, brands, creators)
    assert len(examples) == 2
    assert examples[0]["label"] == 2
    assert examples[1]["label"] == 0
    assert examples[0]["brand_id"] == "b01"
    assert examples[0]["creator_id"] == "c01"


def test_build_training_examples_preserves_ids():
    labels = [
        {"brand_id": "b01", "creator_id": "c01", "label": "maybe"},
    ]
    brands = {"b01": {"brand_id": "b01", "brand_name": "B", "industry": "I"}}
    creators = {"c01": {"creator_id": "c01", "name": "C", "primary_niche": "n"}}
    examples = build_training_examples(labels, brands, creators)
    assert examples[0]["brand_id"] == "b01"
    assert examples[0]["creator_id"] == "c01"
    assert examples[0]["label"] == 1


if __name__ == "__main__":
    test_label_mapping()
    test_split_brands_no_overlap()
    test_split_brands_complete_coverage()
    test_split_deterministic()
    test_split_manifest_structure()
    test_labels_for_split()
    test_build_training_examples()
    test_build_training_examples_preserves_ids()
    print("All tests passed.")
