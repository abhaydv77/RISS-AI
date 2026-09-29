import json
import math
import unittest
from pathlib import Path

from feature_engineering import (
    FEATURE_NAMES,
    FX_PER_USD,
    engineer_features,
    feature_vector,
    parse_age_range,
    parse_rate_card,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FEATURE_NAMES = (
    "niche_primary_match", "niche_secondary_overlap", "niche_preferred_match",
    "audience_age_overlap", "audience_gender_match", "audience_location_overlap",
    "creator_in_target_country", "creator_in_mandatory_region", "platform_coverage",
    "platform_all_covered", "budget_to_min_rate_ratio", "budget_fits_min",
    "followers_percentile_in_range", "followers_log", "creator_size_tier",
    "content_type_jaccard", "has_excluded_trait", "mandatory_requirements_met",
    "engagement_rate", "avg_views_per_follower",
)


def profiles():
    brands = json.loads((ROOT / "data/brands.json").read_text())
    creators = json.loads((ROOT / "data/creators.json").read_text())
    return brands, creators


class ParsingTests(unittest.TestCase):
    def test_age_range_parsing(self):
        self.assertEqual(parse_age_range("18-30"), (18.0, 30.0))
        self.assertEqual(parse_age_range("30–18 years"), (18.0, 30.0))
        self.assertIsNone(parse_age_range("unknown"))

    def test_rate_card_parsing(self):
        self.assertEqual(parse_rate_card("₹15,000–₹50,000"), (15000 / 83, 50000 / 83))
        self.assertEqual(parse_rate_card("$1,200-$2,400"), (1200.0, 2400.0))
        self.assertIsNone(parse_rate_card("negotiable", "INR"))

    def test_currency_conversion_for_budget_and_rate(self):
        brand = {"budget": 83000, "currency": "INR"}
        creator = {"rate_card": "₹41,500–₹83,000"}
        # Both sides convert to USD, so the affordability ratio is 2.
        self.assertAlmostEqual(engineer_features(brand, creator)["budget_to_min_rate_ratio"], math.log1p(2))
        for currency, local_per_usd in FX_PER_USD.items():
            with self.subTest(currency=currency):
                self.assertEqual(parse_rate_card(str(local_per_usd), currency), (1.0, 1.0))


class FeatureBehaviorTests(unittest.TestCase):
    def test_niche_matching(self):
        brand = {"required_creator_niches": ["fitness"], "preferred_creator_niches": ["wellness"]}
        creator = {"primary_niche": "fitness", "secondary_niches": ["wellness"]}
        got = engineer_features(brand, creator)
        self.assertEqual((got["niche_primary_match"], got["niche_secondary_overlap"], got["niche_preferred_match"]), (1, 1, 0))

    def test_audience_matching(self):
        brand = {"target_age_range": "18-30", "target_gender": "female", "target_locations": ["India", "UAE"]}
        creator = {"audience_age_range": "25-34", "audience_gender_distribution": {"female": .72},
                   "audience_locations": ["India", "Singapore"]}
        got = engineer_features(brand, creator)
        self.assertAlmostEqual(got["audience_age_overlap"], 5 / 12)
        self.assertEqual(got["audience_gender_match"], 1)
        self.assertAlmostEqual(got["audience_location_overlap"], 1 / 3)

    def test_geography_matching(self):
        brand = {"target_locations": ["India"], "mandatory_requirements": ["must be based in India"]}
        creator = {"location": "Mumbai, India"}
        got = engineer_features(brand, creator)
        self.assertEqual(got["creator_in_target_country"], 1)
        self.assertEqual(got["creator_in_mandatory_region"], 1)
        self.assertEqual(got["mandatory_requirements_met"], 1)

    def test_platform_coverage(self):
        brand = {"platforms": ["instagram", "youtube", "tiktok"]}
        creator = {"platforms": ["instagram", "youtube"]}
        got = engineer_features(brand, creator)
        self.assertAlmostEqual(got["platform_coverage"], 2 / 3)
        self.assertEqual(got["platform_all_covered"], 0)

    def test_budget_features(self):
        brand = {"budget": 300000, "currency": "INR"}
        creator = {"rate_card": "₹50,000–₹150,000"}
        got = engineer_features(brand, creator)
        self.assertAlmostEqual(got["budget_to_min_rate_ratio"], math.log1p(6))
        self.assertEqual(got["budget_fits_min"], 1)

    def test_follower_and_creator_size_features(self):
        brand = {"minimum_followers": 100000, "maximum_followers": 500000}
        creator = {"followers": 300000}
        got = engineer_features(brand, creator)
        self.assertEqual(got["followers_percentile_in_range"], .5)
        self.assertAlmostEqual(got["followers_log"], math.log10(300001))
        self.assertEqual(got["creator_size_tier"], 1)

    def test_content_type_similarity(self):
        brand = {"content_types": ["workout", "nutrition", "yoga"]}
        creator = {"content_types": ["workout", "nutrition", "wellness"]}
        self.assertAlmostEqual(engineer_features(brand, creator)["content_type_jaccard"], 2 / 4)

    def test_mandatory_requirement_checks(self):
        brand = {"mandatory_requirements": ["must post in English or Hindi", "must have fashion AND fitness content"]}
        creator = {"languages": ["Hindi"], "content_types": ["fashion", "fitness"]}
        self.assertEqual(engineer_features(brand, creator)["mandatory_requirements_met"], 1)
        creator["content_types"] = ["fashion"]
        self.assertEqual(engineer_features(brand, creator)["mandatory_requirements_met"], .5)

    def test_mandatory_audience_gender_requirement(self):
        brand = {"mandatory_requirements": ["must have male-skewed audience"]}
        creator = {"audience_gender_distribution": {"male": .7, "female": .3}}
        self.assertEqual(engineer_features(brand, creator)["mandatory_requirements_met"], 1)

    def test_excluded_trait_checks(self):
        brand = {"excluded_traits": ["male-dominated audience", "luxury-focused"]}
        male_creator = {"audience_gender_distribution": {"male": .8}}
        self.assertEqual(engineer_features(brand, male_creator)["has_excluded_trait"], 1)
        neutral_creator = {"audience_gender_distribution": {"male": .4}, "bio": "Everyday wellness videos"}
        self.assertEqual(engineer_features(brand, neutral_creator)["has_excluded_trait"], 0)
        only_male_brand = {"excluded_traits": ["male-only audience"]}
        self.assertEqual(engineer_features(only_male_brand, male_creator)["has_excluded_trait"], 1)
        female_skew_brand = {"excluded_traits": ["female-skewed audience"]}
        self.assertEqual(engineer_features(female_skew_brand, male_creator)["has_excluded_trait"], 0)


class DatasetAndVectorTests(unittest.TestCase):
    def test_real_dataset_pairs_and_vector_contract(self):
        brands, creators = profiles()
        self.assertGreaterEqual(len(brands), 2)
        self.assertGreaterEqual(len(creators), 2)
        for brand, creator in ((brands[0], creators[0]), (brands[1], creators[4]),
                               (brands[2], creators[7]), (brands[3], creators[2])):
            features = engineer_features(brand, creator)
            vector = feature_vector(brand, creator)
            self.assertEqual(tuple(features), FEATURE_NAMES)
            self.assertEqual(FEATURE_NAMES, EXPECTED_FEATURE_NAMES)
            self.assertEqual(len(FEATURE_NAMES), 20)
            self.assertEqual(len(vector), 20)
            self.assertTrue(all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector))


if __name__ == "__main__":
    unittest.main()
