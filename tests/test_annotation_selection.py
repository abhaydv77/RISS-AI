import json
import unittest
from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_json(name):
    return json.loads((DATA_DIR / name).read_text())


class AnnotationSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brands = load_json("brands.json")
        cls.creators = load_json("creators.json")
        cls.pairs_universe = load_json("pairs_v0.json")
        cls.labels = load_json("labels.json")
        cls.selected = load_json("annotation_pairs_v1.json")

    def test_selection_has_exactly_2000_pairs(self):
        self.assertEqual(len(self.selected["pairs"]), 2000)

    def test_all_400_existing_labeled_pairs_present(self):
        existing = set(
            (l["brand_id"], l["creator_id"]) for l in self.labels["labels"]
        )
        selected = set(
            (p["brand_id"], p["creator_id"]) for p in self.selected["pairs"]
        )
        self.assertTrue(existing.issubset(selected))

    def test_no_duplicate_pairs(self):
        pairs = [(p["brand_id"], p["creator_id"]) for p in self.selected["pairs"]]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_every_pair_exists_in_universe(self):
        universe = set(
            (p["brand_id"], p["creator_id"]) for p in self.pairs_universe["pairs"]
        )
        for p in self.selected["pairs"]:
            self.assertIn((p["brand_id"], p["creator_id"]), universe)

    def test_no_selected_pair_references_invalid_brand(self):
        brand_ids = set(b["brand_id"] for b in self.brands)
        for p in self.selected["pairs"]:
            self.assertIn(p["brand_id"], brand_ids)

    def test_no_selected_pair_references_invalid_creator(self):
        creator_ids = set(c["creator_id"] for c in self.creators)
        for p in self.selected["pairs"]:
            self.assertIn(p["creator_id"], creator_ids)

    def test_exactly_1600_new_unlabeled_pairs(self):
        existing = set(
            (l["brand_id"], l["creator_id"]) for l in self.labels["labels"]
        )
        selected = set(
            (p["brand_id"], p["creator_id"]) for p in self.selected["pairs"]
        )
        new_pairs = selected - existing
        self.assertEqual(len(new_pairs), 1600)

    def test_all_50_brands_represented(self):
        brand_ids = set(b["brand_id"] for b in self.brands)
        selected_brands = set(p["brand_id"] for p in self.selected["pairs"])
        self.assertEqual(brand_ids, selected_brands)

    def test_all_200_creators_represented(self):
        creator_ids = set(c["creator_id"] for c in self.creators)
        selected_creators = set(p["creator_id"] for p in self.selected["pairs"])
        self.assertEqual(creator_ids, selected_creators)

    def test_selection_is_deterministic(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, str(ROOT / "select_annotation_pairs.py")],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        self.assertEqual(result.returncode, 0)

        reloaded = load_json("annotation_pairs_v1.json")
        self.assertEqual(self.selected["pairs"], reloaded["pairs"])

    def test_meta_fields(self):
        meta = self.selected["meta"]
        self.assertEqual(meta["total_pairs"], 2000)
        self.assertEqual(meta["existing_labeled_pairs"], 400)
        self.assertEqual(meta["new_unlabeled_pairs"], 1600)
        self.assertEqual(meta["source_pair_universe"], "data/pairs_v0.json")
        self.assertEqual(meta["schema_version"], "1.0")
        self.assertEqual(meta["selection_method"], "deterministic_stratified_coverage")


if __name__ == "__main__":
    unittest.main()
