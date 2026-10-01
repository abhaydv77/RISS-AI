import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_json(name):
    return json.loads((DATA_DIR / name).read_text())


class PairUniverseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brands = load_json("brands.json")
        cls.creators = load_json("creators.json")
        cls.pairs_doc = load_json("pairs_v0.json")
        cls.labels = load_json("labels.json")

    def test_pair_count_is_cartesian_product(self):
        self.assertEqual(len(self.brands) * len(self.creators), 10000)
        self.assertEqual(len(self.pairs_doc["pairs"]), 10000)

    def test_pair_ids_are_unique(self):
        pair_set = set(
            (p["brand_id"], p["creator_id"]) for p in self.pairs_doc["pairs"]
        )
        self.assertEqual(len(pair_set), len(self.pairs_doc["pairs"]))

    def test_every_pair_references_valid_brand(self):
        brand_ids = set(b["brand_id"] for b in self.brands)
        for pair in self.pairs_doc["pairs"]:
            self.assertIn(pair["brand_id"], brand_ids)

    def test_every_pair_references_valid_creator(self):
        creator_ids = set(c["creator_id"] for c in self.creators)
        for pair in self.pairs_doc["pairs"]:
            self.assertIn(pair["creator_id"], creator_ids)

    def test_pair_ordering_is_deterministic(self):
        expected = [
            (b["brand_id"], c["creator_id"])
            for b in self.brands
            for c in self.creators
        ]
        actual = [
            (p["brand_id"], p["creator_id"]) for p in self.pairs_doc["pairs"]
        ]
        self.assertEqual(expected, actual)

    def test_meta_fields(self):
        meta = self.pairs_doc["meta"]
        self.assertEqual(meta["total_pairs"], 10000)
        self.assertEqual(meta["num_brands"], 50)
        self.assertEqual(meta["num_creators"], 200)
        self.assertEqual(meta["schema_version"], "1.0")

    def test_existing_labels_are_subset_of_pair_universe(self):
        pair_set = set(
            (p["brand_id"], p["creator_id"]) for p in self.pairs_doc["pairs"]
        )
        labeled_pairs = set(
            (l["brand_id"], l["creator_id"]) for l in self.labels["labels"]
        )
        self.assertTrue(labeled_pairs.issubset(pair_set))

    def test_original_400_pairs_preserved(self):
        labeled_pairs = set(
            (l["brand_id"], l["creator_id"]) for l in self.labels["labels"]
        )
        original_pairs = set(
            (b["brand_id"], c["creator_id"])
            for b in self.brands[:10]
            for c in self.creators[:40]
        )
        self.assertEqual(labeled_pairs, original_pairs)


if __name__ == "__main__":
    unittest.main()
