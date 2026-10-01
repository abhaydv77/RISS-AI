#!/usr/bin/env python3
"""Build the full 10,000 pair inventory and balanced 2,000-pair Phase 1 set."""
import json
import random
from pathlib import Path

from expand_dataset import ROOT, add_brands, add_creators, read, write
from generate_annotation_pairs import generate_annotation_pairs
from feature_engineering import engineer_features

def quality(pair):
    s = pair["_helper"]["_suggested"]
    return (s["niche_match"] == "strong") + (s["audience_match"] == "strong") + (s["geography_match"] == "strong") + (s["platform_match"] == "strong") + (s["budget_fit"] == "fit") + (s["creator_size_fit"] == "fit") + (s["campaign_fit"] == "strong")

def feature_score(pair, brands, creators):
    f = engineer_features(brands[pair["brand_id"]], creators[pair["creator_id"]])
    return (2*f["niche_primary_match"] + f["niche_secondary_overlap"] + f["niche_preferred_match"]
            + f["audience_age_overlap"] + f["audience_gender_match"] + f["audience_location_overlap"]
            + f["creator_in_target_country"] + f["platform_coverage"] + f["platform_all_covered"]
            + f["budget_fits_min"] + f["content_type_jaccard"] + f["mandatory_requirements_met"]
            - f["has_excluded_trait"])

def make_pairs():
    pairs = generate_annotation_pairs()
    brands = {b["brand_id"]:b for b in read("brands.json")}
    creators = {c["creator_id"]:c for c in read("creators.json")}
    by_brand = {}
    for p in pairs:
        by_brand.setdefault(p["brand_id"], []).append(p)
    chosen = []
    creator_exposure = {}
    # The specification's seven quotas produce exactly 40 creators per brand.
    quota = [("positive",8),("borderline",12),("hard_negative",12),("geography",4),("platform",2),("budget",1),("size",1)]
    for bid, candidates in by_brand.items():
        left = list(candidates)
        buckets = [
            lambda p: quality(p) >= 6,
            lambda p: 3 <= quality(p) <= 5,
            lambda p: quality(p) <= 2,
            lambda p: p["_helper"]["_suggested"]["geography_match"] == "weak",
            lambda p: p["_helper"]["_suggested"]["platform_match"] == "weak",
            lambda p: p["_helper"]["_suggested"]["budget_fit"] in ("mismatch", "uncertain"),
            lambda p: p["_helper"]["_suggested"]["creator_size_fit"] == "outside",
        ]
        for (category, n), predicate in zip(quota, buckets):
            pool = [p for p in left if predicate(p)]
            pool.sort(key=lambda p: ((-feature_score(p, brands, creators) if category == "positive" else feature_score(p, brands, creators)), creator_exposure.get(p["creator_id"], 0), p["creator_id"]))
            # Fill sparse categories from the least/most similar remaining records.
            if len(pool) < n:
                pool += [p for p in sorted(left, key=lambda p: ((-feature_score(p, brands, creators) if category == "positive" else feature_score(p, brands, creators)), creator_exposure.get(p["creator_id"], 0), p["creator_id"])) if p not in pool]
            selected = pool[:n]
            for p in selected:
                p["_phase1_category"] = category
            chosen.extend(selected)
            for p in selected:
                creator_exposure[p["creator_id"]] = creator_exposure.get(p["creator_id"], 0) + 1
            selected_ids = {p["creator_id"] for p in selected}
            left = [p for p in left if p["creator_id"] not in selected_ids]
    full_meta = {"total_pairs":len(pairs),"num_brands":50,"num_creators":200,"schema_version":"2.0","annotation_guide":"reports/ANNOTATION_GUIDE.md","profiles_synthetic":True}
    write("all_annotation_pairs.json", {"meta":full_meta,"pairs":pairs})
    # Keep annotator-facing file focused on the initial stratified batch.
    write("annotation_pairs.json", {"meta":{**full_meta,"total_pairs":len(chosen),"phase":1,"phase1_categories":{"positive":400,"borderline":600,"hard_negative":600,"geography":200,"platform":100,"budget":50,"size":50}},"pairs":chosen})

def main():
    add_brands()
    add_creators()
    make_pairs()

if __name__ == "__main__":
    main()
