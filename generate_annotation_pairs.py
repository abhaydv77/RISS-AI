#!/usr/bin/env python3
"""
Generate all 400 brand-creator pairs for manual annotation.
Outputs annotation_pairs.json with empty labels ready for human annotation.
"""

import json
import re
from pathlib import Path


def parse_rate_card(rate_card: str) -> tuple[int | None, int | None]:
    """Parse rate card like '₹15,000–₹50,000' -> (15000, 50000)."""
    if not rate_card:
        return None, None
    # Remove currency symbols and commas
    cleaned = rate_card.replace('₹', '').replace('$', '').replace(',', '').replace(' ', '')
    # Find numbers
    nums = re.findall(r'\d+', cleaned)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    elif len(nums) == 1:
        return int(nums[0]), int(nums[0])
    return None, None


def parse_age_range(age_str: str) -> tuple[int, int]:
    """Parse '18-30' -> (18, 30)."""
    parts = age_str.split('-')
    return int(parts[0]), int(parts[1])


def age_overlap(brand_range: str, creator_range: str) -> float:
    """Return overlap ratio 0-1."""
    b_min, b_max = parse_age_range(brand_range)
    c_min, c_max = parse_age_range(creator_range)
    overlap_min = max(b_min, c_min)
    overlap_max = min(b_max, c_max)
    if overlap_min >= overlap_max:
        return 0.0
    overlap = overlap_max - overlap_min
    brand_span = b_max - b_min
    creator_span = c_max - c_min
    return overlap / min(brand_span, creator_span)


def gender_match(target_gender: str, audience_dist: dict) -> bool:
    """Check if audience gender matches target."""
    if target_gender == "all":
        return True
    if target_gender == "female":
        return audience_dist.get("female", 0) >= 0.6
    if target_gender == "male":
        return audience_dist.get("male", 0) >= 0.6
    return True


def location_overlap(creator_locs: list, brand_locs: list) -> bool:
    """Any country in common."""
    return bool(set(creator_locs) & set(brand_locs))


def creator_base_country(location: str) -> str:
    """Extract country from 'City, Country'."""
    return location.split(', ')[-1]


def niche_match_level(creator, brand) -> str:
    """Determine niche_match level."""
    creator_niches = {creator["primary_niche"]} | set(creator["secondary_niches"])
    required = set(brand["required_creator_niches"])
    preferred = set(brand["preferred_creator_niches"])
    all_brand_niches = required | preferred

    if creator["primary_niche"] in required:
        return "strong"
    if creator["primary_niche"] in preferred or (creator_niches & required):
        return "partial"
    if creator_niches & all_brand_niches:
        return "partial"
    return "weak"


def audience_match_level(creator, brand) -> str:
    """Determine audience_match level."""
    age_score = age_overlap(brand["target_age_range"], creator["audience_age_range"])
    gender_ok = gender_match(brand["target_gender"], creator["audience_gender_distribution"])
    loc_ok = location_overlap(creator["audience_locations"], brand["target_locations"])

    matches = sum([age_score > 0.3, gender_ok, loc_ok])
    if matches == 3:
        return "strong"
    if matches == 2:
        return "partial"
    return "weak"


def geography_match_level(creator, brand) -> str:
    """Determine geography_match level."""
    creator_country = creator_base_country(creator["location"])
    target = set(brand["target_locations"])

    # Check mandatory location requirements
    mandatory_locs = []
    for req in brand["mandatory_requirements"]:
        req_lower = req.lower()
        if "based in" in req_lower or "location" in req_lower:
            # Extract country mentions (simplified)
            for c in ["India", "UAE", "Singapore", "South Korea", "Japan", "UK", "US", "France", "Germany", "Sweden", "Norway", "Denmark", "Italy", "Spain", "Australia", "Canada"]:
                if c.lower() in req_lower:
                    mandatory_locs.append(c)

    if creator_country in target:
        if mandatory_locs and creator_country not in mandatory_locs:
            return "partial"
        return "strong"
    if location_overlap(creator["audience_locations"], brand["target_locations"]):
        return "partial"
    return "weak"


def platform_match_level(creator, brand) -> str:
    """Determine platform_match level."""
    creator_platforms = set(creator["platforms"])
    brand_platforms = set(brand["platforms"])

    if brand_platforms.issubset(creator_platforms):
        return "strong"
    if brand_platforms & creator_platforms:
        return "partial"
    return "weak"


def budget_fit_level(creator, brand) -> str:
    """Determine budget_fit level."""
    creator_min, _ = parse_rate_card(creator["rate_card"])
    brand_budget = brand["budget"]

    if creator_min is None:
        return "uncertain"

    # Simple comparison (same currency assumed for now; annotator should note currency)
    if brand_budget >= creator_min:
        return "fit"
    if brand_budget >= creator_min * 0.5:
        return "uncertain"
    return "mismatch"


def creator_size_fit_level(creator, brand) -> str:
    """Determine creator_size_fit level."""
    followers = creator["followers"]
    min_f = brand["minimum_followers"]
    max_f = brand["maximum_followers"]

    if min_f <= followers <= max_f:
        return "fit"
    # Near boundary (within 20%)
    if followers >= min_f * 0.8 and followers < min_f:
        return "near"
    if followers <= max_f * 1.2 and followers > max_f:
        return "near"
    return "outside"


def campaign_fit_level(creator, brand) -> str:
    """Determine campaign_fit level."""
    # Content type overlap
    creator_content = set(creator["content_types"])
    brand_content = set(brand["content_types"])
    content_overlap = len(creator_content & brand_content) / max(len(brand_content), 1)

    # Style/tone match (simple keyword check)
    style = creator["content_style"].lower()
    tone = brand["tone"].lower()
    style_match = any(w in style for w in tone.split()) or any(w in tone for w in style.split())

    # Excluded traits check
    excluded_violation = False
    for trait in brand["excluded_traits"]:
        trait_lower = trait.lower()
        # Check creator niches, bio, content_style
        if (trait_lower in creator["primary_niche"].lower() or
            any(trait_lower in n.lower() for n in creator["secondary_niches"]) or
            trait_lower in creator["content_style"].lower() or
            trait_lower in creator["bio"].lower()):
            excluded_violation = True
            break

    # Mandatory requirements check
    mandatory_met = True
    for req in brand["mandatory_requirements"]:
        req_lower = req.lower()
        # Simple checks
        if "based in" in req_lower:
            countries = ["India", "UAE", "Singapore", "South Korea", "Japan", "UK", "US", "France", "Germany", "Sweden", "Norway", "Denmark", "Italy", "Spain", "Australia", "Canada"]
            if not any(c.lower() in req_lower and c.lower() in creator["location"].lower() for c in countries):
                mandatory_met = False
                break
        if "english" in req_lower and "English" not in creator["languages"]:
            mandatory_met = False
            break
        if "hindi" in req_lower and "Hindi" not in creator["languages"]:
            mandatory_met = False
            break
        if "fitness content" in req_lower and "fitness" not in creator["primary_niche"] and "fitness" not in creator["secondary_niches"]:
            mandatory_met = False
            break
        if "skincare content" in req_lower and "skincare" not in creator["primary_niche"] and "skincare" not in creator["secondary_niches"]:
            mandatory_met = False
            break
        if "gaming content" in req_lower and "gaming" not in creator["primary_niche"] and "gaming" not in creator["secondary_niches"]:
            mandatory_met = False
            break
        if "parenting content" in req_lower and "parenting" not in creator["primary_niche"] and "parenting" not in creator["secondary_niches"]:
            mandatory_met = False
            break
        if "fashion" in req_lower and "fashion" not in creator["primary_niche"] and "fashion" not in creator["secondary_niches"]:
            mandatory_met = False
            break
        if "luxury" in req_lower and "luxury" not in creator["primary_niche"] and "luxury" not in creator["secondary_niches"]:
            mandatory_met = False
            break
        if "video" in req_lower and not any(p in ["youtube", "tiktok", "instagram"] for p in creator["platforms"]):
            mandatory_met = False
            break

    if excluded_violation or not mandatory_met:
        return "weak"
    if content_overlap > 0.5 and style_match:
        return "strong"
    if content_overlap > 0.2 or style_match:
        return "partial"
    return "weak"


def generate_annotation_pairs():
    """Generate all 400 brand-creator pairs with empty annotations."""
    with open("data/brands.json") as f:
        brands = json.load(f)
    with open("data/creators.json") as f:
        creators = json.load(f)

    pairs = []
    for brand in brands:
        for creator in creators:
            pair = {
                "brand_id": brand["brand_id"],
                "creator_id": creator["creator_id"],
                "label": "",  # To be filled: good | maybe | poor
                "reasoning": {
                    "niche_match": "",      # strong | partial | weak
                    "audience_match": "",   # strong | partial | weak
                    "geography_match": "",  # strong | partial | weak
                    "platform_match": "",   # strong | partial | weak
                    "budget_fit": "",       # fit | uncertain | mismatch
                    "creator_size_fit": "", # fit | near | outside
                    "campaign_fit": ""      # strong | partial | weak
                },
                "reason": "",  # Short explanation
                # Helper fields for annotator reference (not part of final labels.json)
                "_helper": {
                    "brand_name": brand["brand_name"],
                    "creator_name": creator["name"],
                    "brand_industry": brand["industry"],
                    "creator_primary_niche": creator["primary_niche"],
                    "brand_required_niches": brand["required_creator_niches"],
                    "brand_preferred_niches": brand["preferred_creator_niches"],
                    "creator_secondary_niches": creator["secondary_niches"],
                    "brand_target_age": brand["target_age_range"],
                    "creator_audience_age": creator["audience_age_range"],
                    "brand_target_gender": brand["target_gender"],
                    "creator_audience_gender": creator["audience_gender_distribution"],
                    "brand_target_locations": brand["target_locations"],
                    "creator_audience_locations": creator["audience_locations"],
                    "creator_base_location": creator["location"],
                    "brand_platforms": brand["platforms"],
                    "creator_platforms": creator["platforms"],
                    "brand_budget": brand["budget"],
                    "brand_currency": brand["currency"],
                    "creator_rate_card": creator["rate_card"],
                    "brand_min_followers": brand["minimum_followers"],
                    "brand_max_followers": brand["maximum_followers"],
                    "creator_followers": creator["followers"],
                    "brand_content_types": brand["content_types"],
                    "creator_content_types": creator["content_types"],
                    "brand_tone": brand["tone"],
                    "creator_content_style": creator["content_style"],
                    "brand_mandatory": brand["mandatory_requirements"],
                    "brand_excluded": brand["excluded_traits"],
                    "creator_bio": creator["bio"],
                    # Pre-computed suggestions (annotator can override)
                    "_suggested": {
                        "niche_match": niche_match_level(creator, brand),
                        "audience_match": audience_match_level(creator, brand),
                        "geography_match": geography_match_level(creator, brand),
                        "platform_match": platform_match_level(creator, brand),
                        "budget_fit": budget_fit_level(creator, brand),
                        "creator_size_fit": creator_size_fit_level(creator, brand),
                        "campaign_fit": campaign_fit_level(creator, brand)
                    }
                }
            }
            pairs.append(pair)

    return pairs


def main():
    pairs = generate_annotation_pairs()

    output = {
        "meta": {
            "total_pairs": len(pairs),
            "num_brands": 10,
            "num_creators": 40,
            "schema_version": "1.0",
            "annotation_guide": "ANNOTATION_GUIDE.md"
        },
        "pairs": pairs
    }

    with open("annotation_pairs.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(pairs)} pairs -> annotation_pairs.json")
    print("Open annotation_pairs.json and fill in label + reasoning for each pair.")
    print("Then save completed file as data/labels.json")


if __name__ == "__main__":
    main()