#!/usr/bin/env python3
"""Deterministic stratified selection of ~2,000 pairs for ground-truth annotation.

Selection strategy:
1. Preserve all 400 existing labeled pairs.
2. Select ~1,600 additional pairs using coverage-based sampling across:
   - niche overlap (strong / partial / weak)
   - geography overlap (strong / partial / weak)
   - platform overlap (full / partial / none)
   - creator size fit (in-range / near / outside)
   - brand target gender
   - creator primary niche
   - creator geography
3. Ensure every brand and creator is represented.
4. Deterministic: same input always produces same output.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def load_json(name):
    return json.loads((DATA_DIR / name).read_text())


def get_country(location):
    return location.split(",")[-1].strip() if "," in location else location.strip()


def get_size_bucket(followers):
    if followers < 50000:
        return "micro"
    elif followers < 500000:
        return "mid-tier"
    elif followers < 2000000:
        return "macro"
    else:
        return "mega"


def parse_rate_card_min(rate_card):
    import re
    numbers = re.findall(r"[\d,]+", rate_card.replace("₹", "").replace(",", "").replace("$", "").replace(" ", ""))
    if numbers:
        return int(numbers[0])
    return 0


def compute_selection_signals(brand, creator):
    """Compute lightweight deterministic signals for stratification."""
    brand_niches = set(brand.get("required_creator_niches", [])) | set(brand.get("preferred_creator_niches", []))
    creator_niches = set(creator.get("secondary_niches", [])) | {creator.get("primary_niche", "")}

    niche_overlap = "strong" if creator.get("primary_niche", "") in brand.get("required_creator_niches", []) else \
                     "partial" if creator_niches & brand_niches else "weak"

    brand_locs = set(brand.get("target_locations", []))
    creator_locs = set(creator.get("audience_locations", []))
    creator_country = get_country(creator.get("location", ""))

    if creator_country in brand_locs:
        geo_overlap = "strong"
    elif creator_locs & brand_locs:
        geo_overlap = "partial"
    else:
        geo_overlap = "weak"

    brand_platforms = set(brand.get("platforms", []))
    creator_platforms = set(creator.get("platforms", []))

    if brand_platforms <= creator_platforms:
        platform_overlap = "full"
    elif brand_platforms & creator_platforms:
        platform_overlap = "partial"
    else:
        platform_overlap = "none"

    followers = creator.get("followers", 0)
    min_f = brand.get("minimum_followers", 0)
    max_f = brand.get("maximum_followers", 10000000)

    if min_f <= followers <= max_f:
        size_fit = "in-range"
    elif min_f * 0.8 <= followers <= max_f * 1.2:
        size_fit = "near"
    else:
        size_fit = "outside"

    budget = brand.get("budget", 0)
    rate_min = parse_rate_card_min(creator.get("rate_card", "0"))
    if rate_min > 0 and budget >= rate_min:
        budget_fit = "fit"
    elif rate_min > 0 and budget >= rate_min * 0.5:
        budget_fit = "uncertain"
    else:
        budget_fit = "mismatch"

    return {
        "niche_overlap": niche_overlap,
        "geo_overlap": geo_overlap,
        "platform_overlap": platform_overlap,
        "size_fit": size_fit,
        "budget_fit": budget_fit,
        "creator_niche": creator.get("primary_niche", ""),
        "creator_country": creator_country,
        "creator_size_bucket": get_size_bucket(followers),
        "brand_industry": brand.get("industry", ""),
        "brand_target_gender": brand.get("target_gender", "all"),
    }


def select_pairs():
    brands = load_json("brands.json")
    creators = load_json("creators.json")
    pairs_universe = load_json("pairs_v0.json")
    labels = load_json("labels.json")

    brand_map = {b["brand_id"]: b for b in brands}
    creator_map = {c["creator_id"]: c for c in creators}

    existing_labeled = set((l["brand_id"], l["creator_id"]) for l in labels["labels"])
    all_pairs = set((p["brand_id"], p["creator_id"]) for p in pairs_universe["pairs"])
    unlabeled_candidates = sorted(all_pairs - existing_labeled)

    selected = set(existing_labeled)

    signals = {}
    for bid, cid in unlabeled_candidates:
        signals[(bid, cid)] = compute_selection_signals(brand_map[bid], creator_map[cid])

    creator_pair_count = Counter()
    for bid, cid in selected:
        creator_pair_count[cid] += 1

    brand_pair_count = Counter()
    for bid, cid in selected:
        brand_pair_count[bid] += 1

    underrepresented_creators = [cid for cid in creator_map if creator_pair_count[cid] < 5]
    underrepresented_creators.sort(key=lambda cid: creator_pair_count[cid])

    for cid in underrepresented_creators:
        candidates = [(bid, c) for (bid, c) in unlabeled_candidates if c == cid and (bid, c) not in selected]
        candidates.sort(key=lambda x: (brand_pair_count[x[0]], x[0]))
        for bid, c in candidates:
            if creator_pair_count[cid] >= 5:
                break
            selected.add((bid, c))
            creator_pair_count[cid] += 1
            brand_pair_count[bid] += 1

    underrepresented_brands = [bid for bid in brand_map if brand_pair_count[bid] < 30]
    underrepresented_brands.sort(key=lambda bid: brand_pair_count[bid])

    for bid in underrepresented_brands:
        candidates = [(b, cid) for (b, cid) in unlabeled_candidates if b == bid and (b, cid) not in selected]
        candidates.sort(key=lambda x: (creator_pair_count[x[1]], x[1]))
        for b, cid in candidates:
            if brand_pair_count[bid] >= 30:
                break
            selected.add((b, cid))
            creator_pair_count[cid] += 1
            brand_pair_count[bid] += 1

    remaining_needed = 2000 - len(selected)
    if remaining_needed > 0:
        remaining = [(bid, cid) for (bid, cid) in unlabeled_candidates if (bid, cid) not in selected]

        strata = defaultdict(list)
        for pair in remaining:
            s = signals[pair]
            key = (s["niche_overlap"], s["geo_overlap"], s["platform_overlap"], s["size_fit"])
            strata[key].append(pair)

        stratum_keys = sorted(strata.keys())
        idx = 0
        while remaining_needed > 0 and stratum_keys:
            key = stratum_keys[idx % len(stratum_keys)]
            if strata[key]:
                pair = strata[key].pop(0)
                selected.add(pair)
                creator_pair_count[pair[1]] += 1
                brand_pair_count[pair[0]] += 1
                remaining_needed -= 1
            idx += 1
            if not any(strata[k] for k in stratum_keys):
                break

    if len(selected) != 2000:
        raise RuntimeError(f"Selection produced {len(selected)} pairs, expected 2000")

    ordered_selected = [(bid, cid) for bid in brand_map for cid in creator_map if (bid, cid) in selected]

    return ordered_selected, existing_labeled, unlabeled_candidates, signals, brand_map, creator_map


def build_report(selected, existing_labeled, unlabeled_candidates, signals, brand_map, creator_map):
    selected = set(selected)
    brand_ids = set(bid for bid, _ in selected)
    creator_ids = set(cid for _, cid in selected)

    creator_niche_counts = Counter()
    creator_country_counts = Counter()
    creator_size_counts = Counter()
    brand_industry_counts = Counter()
    brand_gender_counts = Counter()
    platform_counts = Counter()

    for bid, cid in selected:
        c = creator_map[cid]
        b = brand_map[bid]
        creator_niche_counts[c["primary_niche"]] += 1
        creator_country_counts[get_country(c["location"])] += 1
        creator_size_counts[get_size_bucket(c["followers"])] += 1
        brand_industry_counts[b["industry"]] += 1
        brand_gender_counts[b["target_gender"]] += 1
        for p in c["platforms"]:
            platform_counts[p] += 1

    niche_overlap_counts = Counter()
    geo_overlap_counts = Counter()
    platform_overlap_counts = Counter()
    size_fit_counts = Counter()
    budget_fit_counts = Counter()

    for pair in selected:
        if pair in signals:
            s = signals[pair]
            niche_overlap_counts[s["niche_overlap"]] += 1
            geo_overlap_counts[s["geo_overlap"]] += 1
            platform_overlap_counts[s["platform_overlap"]] += 1
            size_fit_counts[s["size_fit"]] += 1
            budget_fit_counts[s["budget_fit"]] += 1

    existing_signals = {pair: signals[pair] for pair in existing_labeled if pair in signals}
    existing_creator_niches = Counter(creator_map[cid]["primary_niche"] for _, cid in existing_labeled)
    existing_creator_countries = Counter(get_country(creator_map[cid]["location"]) for _, cid in existing_labeled)
    existing_creator_sizes = Counter(get_size_bucket(creator_map[cid]["followers"]) for _, cid in existing_labeled)

    new_pairs = set(selected) - existing_labeled
    new_signals = {pair: signals[pair] for pair in new_pairs if pair in signals}
    new_creator_niches = Counter(creator_map[cid]["primary_niche"] for _, cid in new_pairs)
    new_creator_countries = Counter(get_country(creator_map[cid]["location"]) for _, cid in new_pairs)
    new_creator_sizes = Counter(get_size_bucket(creator_map[cid]["followers"]) for _, cid in new_pairs)

    return {
        "phase": "2.4",
        "description": "Stratified selection of ~2,000 pairs for ground-truth annotation",
        "counts": {
            "total_pair_universe": 10000,
            "existing_labeled_pairs": len(existing_labeled),
            "unlabeled_candidates": len(unlabeled_candidates),
            "selected_existing_pairs": len(existing_labeled & selected),
            "selected_new_pairs": len(new_pairs),
            "final_selected_pairs": len(selected),
        },
        "coverage": {
            "brands_represented": len(brand_ids),
            "creators_represented": len(creator_ids),
            "creator_niche_counts": dict(creator_niche_counts.most_common()),
            "creator_country_counts": dict(creator_country_counts.most_common()),
            "creator_size_counts": dict(creator_size_counts.most_common()),
            "brand_industry_counts": dict(brand_industry_counts.most_common()),
            "brand_target_gender_counts": dict(brand_gender_counts.most_common()),
            "platform_counts": dict(platform_counts.most_common()),
        },
        "interaction_coverage": {
            "niche_overlap": dict(niche_overlap_counts),
            "geography_overlap": dict(geo_overlap_counts),
            "platform_overlap": dict(platform_overlap_counts),
            "follower_range_fit": dict(size_fit_counts),
            "budget_fit": dict(budget_fit_counts),
        },
        "existing_labeled_distribution": {
            "count": len(existing_labeled),
            "brands_represented": len(set(bid for bid, _ in existing_labeled)),
            "creators_represented": len(set(cid for _, cid in existing_labeled)),
            "creator_niche_counts": dict(existing_creator_niches.most_common()),
            "creator_country_counts": dict(existing_creator_countries.most_common()),
            "creator_size_counts": dict(existing_creator_sizes.most_common()),
        },
        "new_unlabeled_distribution": {
            "count": len(new_pairs),
            "brands_represented": len(set(bid for bid, _ in new_pairs)),
            "creators_represented": len(set(cid for _, cid in new_pairs)),
            "creator_niche_counts": dict(new_creator_niches.most_common()),
            "creator_country_counts": dict(new_creator_countries.most_common()),
            "creator_size_counts": dict(new_creator_sizes.most_common()),
        },
        "reproducibility": {
            "selection_method": "deterministic_stratified_coverage",
            "seed": None,
            "config": {
                "target_total": 2000,
                "existing_labeled": 400,
                "new_unlabeled": 1600,
                "min_creator_pairs": 5,
                "min_brand_pairs": 30,
            },
            "source_datasets": [
                "data/brands.json",
                "data/creators.json",
                "data/pairs_v0.json",
                "data/labels.json",
            ],
        },
    }


def main():
    selected, existing_labeled, unlabeled_candidates, signals, brand_map, creator_map = select_pairs()

    output = {
        "meta": {
            "total_pairs": len(selected),
            "existing_labeled_pairs": len(existing_labeled),
            "new_unlabeled_pairs": len(selected) - len(existing_labeled),
            "source_pair_universe": "data/pairs_v0.json",
            "schema_version": "1.0",
            "selection_method": "deterministic_stratified_coverage",
        },
        "pairs": [{"brand_id": bid, "creator_id": cid} for bid, cid in selected],
    }

    output_path = DATA_DIR / "annotation_pairs_v1.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {len(selected)} pairs to {output_path}")

    report = build_report(selected, existing_labeled, unlabeled_candidates, signals, brand_map, creator_map)
    report_path = ROOT / "reports" / "phase2_4_selection_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote selection report to {report_path}")


if __name__ == "__main__":
    main()
