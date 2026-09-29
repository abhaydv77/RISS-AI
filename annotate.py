#!/usr/bin/env python3
"""
Riss-AI Phase 1.2: Ground-truth dataset annotation
Annotates 400 brand-creator pairs across 7 dimensions
"""

import json
import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

@dataclass
class DimensionScore:
    level: str  # strong/partial/weak or fit/uncertain/mismatch or fit/near/outside
    reason: str

def parse_rate_card(rate_card: str) -> Tuple[int, int]:
    """Parse rate card like '₹15,000–₹50,000' -> (15000, 50000)"""
    numbers = re.findall(r'[\d,]+', rate_card.replace('₹', '').replace(',', ''))
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    elif len(numbers) == 1:
        val = int(numbers[0])
        return val, val
    return 0, 0

def parse_age_range(age_str: str) -> Tuple[int, int]:
    """Parse age range like '18-30' -> (18, 30)"""
    parts = age_str.split('-')
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    return 0, 100

def age_overlap(brand_age: str, creator_age: str) -> str:
    """Return 'strong', 'partial', or 'weak' based on age range overlap"""
    b_min, b_max = parse_age_range(brand_age)
    c_min, c_max = parse_age_range(creator_age)
    
    overlap_min = max(b_min, c_min)
    overlap_max = min(b_max, c_max)
    
    if overlap_min > overlap_max:
        return "weak"
    
    overlap_size = overlap_max - overlap_min
    brand_size = b_max - b_min
    creator_size = c_max - c_min
    
    if overlap_size >= min(brand_size, creator_size) * 0.8:
        return "strong"
    elif overlap_size > 0:
        return "partial"
    return "weak"

def gender_match(brand_gender: str, creator_gender: Dict) -> bool:
    """Check if gender distribution matches target"""
    if brand_gender == "all":
        return True
    elif brand_gender == "female":
        return creator_gender.get("female", 0) >= 0.6
    elif brand_gender == "male":
        return creator_gender.get("male", 0) >= 0.6
    return False

def location_overlap(brand_locs: List[str], creator_locs: List[str]) -> bool:
    """Check if any location overlaps"""
    return bool(set(brand_locs) & set(creator_locs))

def evaluate_niche_match(brand: Dict, creator: Dict) -> DimensionScore:
    primary = creator.get("primary_niche", "").lower()
    secondary = [n.lower() for n in creator.get("secondary_niches", [])]
    required = [n.lower() for n in brand.get("required_creator_niches", [])]
    preferred = [n.lower() for n in brand.get("preferred_creator_niches", [])]
    
    if primary in required:
        return DimensionScore("strong", f"Primary niche '{primary}' matches required niches {required}")
    elif primary in preferred or any(n in required + preferred for n in secondary):
        return DimensionScore("partial", f"Primary niche '{primary}' in preferred or secondary overlaps")
    return DimensionScore("weak", f"No niche overlap: primary='{primary}', secondary={secondary}, required={required}, preferred={preferred}")

def evaluate_audience_match(brand: Dict, creator: Dict) -> DimensionScore:
    age_match = age_overlap(brand.get("target_age_range", ""), creator.get("audience_age_range", ""))
    gender_ok = gender_match(brand.get("target_gender", "all"), creator.get("audience_gender_distribution", {}))
    location_ok = location_overlap(brand.get("target_locations", []), creator.get("audience_locations", []))
    
    matches = sum([age_match == "strong", gender_ok, location_ok])
    partials = sum([age_match == "partial"])
    
    if matches >= 2 and age_match == "strong":
        return DimensionScore("strong", f"Age: {age_match}, Gender: {'match' if gender_ok else 'mismatch'}, Location: {'overlap' if location_ok else 'no overlap'}")
    elif matches + partials >= 2:
        return DimensionScore("partial", f"Age: {age_match}, Gender: {'match' if gender_ok else 'mismatch'}, Location: {'overlap' if location_ok else 'no overlap'}")
    return DimensionScore("weak", f"Age: {age_match}, Gender: {'match' if gender_ok else 'mismatch'}, Location: {'overlap' if location_ok else 'no overlap'}")

def evaluate_geography_match(brand: Dict, creator: Dict) -> DimensionScore:
    target_locs = brand.get("target_locations", [])
    base_loc = creator.get("location", "")
    base_country = base_loc.split(",")[-1].strip() if "," in base_loc else base_loc
    audience_locs = creator.get("audience_locations", [])
    
    mandatory = brand.get("mandatory_requirements", [])
    location_mandatory = any("based in" in m.lower() or "location" in m.lower() for m in mandatory)
    
    if base_country in target_locs or (location_mandatory and base_country in target_locs):
        return DimensionScore("strong", f"Creator based in {base_country} which is in target locations {target_locs}")
    elif location_overlap(target_locs, audience_locs):
        return DimensionScore("partial", f"Creator based in {base_country} (outside target) but audience overlaps: {set(target_locs) & set(audience_locs)}")
    return DimensionScore("weak", f"Creator based in {base_country}, audience in {audience_locs}, target: {target_locs} - no overlap")

def evaluate_platform_match(brand: Dict, creator: Dict) -> DimensionScore:
    brand_platforms = set(brand.get("platforms", []))
    creator_platforms = set(creator.get("platforms", []))
    
    if brand_platforms.issubset(creator_platforms):
        return DimensionScore("strong", f"All brand platforms {brand_platforms} covered by creator {creator_platforms}")
    elif brand_platforms & creator_platforms:
        return DimensionScore("partial", f"Partial overlap: brand {brand_platforms}, creator {creator_platforms}")
    return DimensionScore("weak", f"No platform overlap: brand {brand_platforms}, creator {creator_platforms}")

def evaluate_budget_fit(brand: Dict, creator: Dict) -> DimensionScore:
    budget = brand.get("budget", 0)
    rate_card = creator.get("rate_card", "0")
    min_rate, max_rate = parse_rate_card(rate_card)
    
    if budget >= min_rate:
        return DimensionScore("fit", f"Budget {budget} >= creator min rate {min_rate}")
    elif budget >= min_rate / 2:
        return DimensionScore("uncertain", f"Budget {budget} within 2x of creator min rate {min_rate}")
    return DimensionScore("mismatch", f"Budget {budget} < creator min rate {min_rate} / 2")

def evaluate_creator_size_fit(brand: Dict, creator: Dict) -> DimensionScore:
    followers = creator.get("followers", 0)
    min_f = brand.get("minimum_followers", 0)
    max_f = brand.get("maximum_followers", 10000000)
    
    if min_f <= followers <= max_f:
        return DimensionScore("fit", f"Followers {followers} within range [{min_f}, {max_f}]")
    elif followers >= min_f * 0.8 and followers <= max_f * 1.2:
        return DimensionScore("near", f"Followers {followers} near boundary of range [{min_f}, {max_f}]")
    return DimensionScore("outside", f"Followers {followers} far outside range [{min_f}, {max_f}]")

def evaluate_campaign_fit(brand: Dict, creator: Dict) -> DimensionScore:
    brand_content = set(brand.get("content_types", []))
    creator_content = set(creator.get("content_types", []))
    content_overlap = len(brand_content & creator_content) > 0
    
    tone = brand.get("tone", "").lower()
    style = creator.get("content_style", "").lower()
    tone_match = any(word in style for word in tone.split()) or any(word in tone for word in style.split())
    
    excluded = brand.get("excluded_traits", [])
    has_excluded = any(excl.lower() in creator.get("bio", "").lower() or 
                       excl.lower() in " ".join(creator.get("content_types", [])).lower() 
                       for excl in excluded)
    
    mandatory = brand.get("mandatory_requirements", [])
    mandatory_met = True
    mandatory_fail_reasons = []
    target_locs = [l.lower() for l in brand.get("target_locations", [])]
    for req in mandatory:
        req_lower = req.lower()
        if "based in" in req_lower:
            # Check if creator's base country is in brand's target locations
            base = creator.get("location", "").split(",")[-1].strip().lower()
            if base not in target_locs:
                mandatory_met = False
                mandatory_fail_reasons.append(f"location: {req} (creator in {base}, target: {target_locs})")
        elif "post in" in req_lower or "language" in req_lower or "speak" in req_lower:
            langs = [l.lower() for l in creator.get("languages", [])]
            req_langs = req_lower.replace("must post in ", "").replace("must be ", "").replace("must speak ", "").split(" or ")
            req_langs = [rl.strip().lower() for rl in req_langs]
            if not any(any(rl in lang for lang in langs) for rl in req_langs):
                mandatory_met = False
                mandatory_fail_reasons.append(f"language: {req}")
        elif "content" in req_lower or "create" in req_lower:
            if not content_overlap:
                mandatory_met = False
                mandatory_fail_reasons.append(f"content type: {req}")
        elif "male" in req_lower or "female" in req_lower:
            target_gender = brand.get("target_gender", "all")
            creator_gender = creator.get("audience_gender_distribution", {})
            if target_gender == "female" and creator_gender.get("female", 0) < 0.6:
                mandatory_met = False
                mandatory_fail_reasons.append(f"audience gender: {req}")
            elif target_gender == "male" and creator_gender.get("male", 0) < 0.6:
                mandatory_met = False
                mandatory_fail_reasons.append(f"audience gender: {req}")
    
    mandatory_str = "mandatory met" if mandatory_met else f"mandatory unmet: {'; '.join(mandatory_fail_reasons)}"
    
    if content_overlap and tone_match and not has_excluded and mandatory_met:
        return DimensionScore("strong", f"Content overlap: {brand_content & creator_content}, tone matches, no excluded traits, {mandatory_str}")
    elif content_overlap or tone_match:
        issues = []
        if has_excluded:
            issues.append("excluded traits present")
        if not mandatory_met:
            issues.append(mandatory_str)
        if not content_overlap:
            issues.append("no content overlap")
        if not tone_match:
            issues.append("tone mismatch")
        return DimensionScore("partial", f"Some match: {'; '.join(issues) if issues else 'minor gaps'}")
    return DimensionScore("weak", f"Content mismatch, tone clash, or excluded traits: {excluded}; {mandatory_str}")

def determine_label(dimensions: Dict[str, DimensionScore]) -> str:
    """Determine overall label from dimension scores"""
    levels = {k: v.level for k, v in dimensions.items()}
    
    # Check hard constraints first
    campaign = dimensions.get("campaign_fit", DimensionScore("weak", ""))
    mandatory_unmet = "mandatory unmet" in campaign.reason.lower()
    has_excluded = "excluded traits present" in campaign.reason.lower()
    
    geo = dimensions.get("geography_match", DimensionScore("weak", ""))
    geo_weak = geo.level == "weak"
    
    niche = dimensions.get("niche_match", DimensionScore("weak", ""))
    niche_weak = niche.level == "weak"
    
    # Hard constraint: excluded traits -> poor
    if has_excluded:
        return "poor"
    
    # Hard constraint: mandatory requirements unmet -> at best maybe
    if mandatory_unmet:
        # Count strong dimensions to see if maybe is warranted
        strong_count = sum(1 for v in levels.values() if v in ["strong", "fit"])
        if strong_count >= 5:
            return "maybe"
        return "poor"
    
    # Hard constraint: geography weak (wrong location for local campaign) -> at best maybe
    if geo_weak:
        strong_count = sum(1 for v in levels.values() if v in ["strong", "fit"])
        if strong_count >= 5:
            return "maybe"
        return "poor"
    
    # Hard constraint: niche weak -> at best maybe
    if niche_weak:
        strong_count = sum(1 for v in levels.values() if v in ["strong", "fit"])
        if strong_count >= 5:
            return "maybe"
        return "poor"
    
    # Count strong/fit, partial/near/uncertain, weak/mismatch/outside
    strong_count = sum(1 for v in levels.values() if v in ["strong", "fit"])
    partial_count = sum(1 for v in levels.values() if v in ["partial", "near", "uncertain"])
    weak_count = sum(1 for v in levels.values() if v in ["weak", "mismatch", "outside"])
    
    # General rules
    if weak_count >= 3:
        return "poor"
    elif weak_count >= 2:
        return "maybe"
    elif strong_count >= 5:
        return "good"
    elif strong_count >= 4:
        return "good"
    elif strong_count >= 3:
        return "maybe"
    else:
        return "maybe"

def format_reason(dimensions: Dict[str, DimensionScore], label: str) -> str:
    """Generate short factual reason"""
    parts = []
    for dim, score in dimensions.items():
        if score.level in ["weak", "mismatch", "outside"]:
            parts.append(f"{dim}: {score.reason}")
    if not parts:
        for dim, score in dimensions.items():
            if score.level in ["partial", "near", "uncertain"]:
                parts.append(f"{dim}: {score.reason}")
                break
    if not parts:
        parts.append("Strong alignment across all dimensions")
    
    return f"{label.capitalize()} fit. " + "; ".join(parts[:3])

def main():
    # Load data
    with open("data/annotation_pairs.json", "r") as f:
        data = json.load(f)
    
    with open("data/creators.json", "r") as f:
        creators_data = json.load(f)
    
    pairs = data["pairs"]
    print(f"Loaded {len(pairs)} pairs to annotate")
    
    annotated = []
    label_counts = {"good": 0, "maybe": 0, "poor": 0}
    ambiguous = []
    
    for i, pair in enumerate(pairs):
        brand_id = pair["brand_id"]
        creator_id = pair["creator_id"]
        helper = pair["_helper"]
        
        # Find full creator data for languages
        creator_full = next((c for c in creators_data if c["creator_id"] == creator_id), {})
        
        # Reconstruct brand and creator objects from helper
        brand = {
            "required_creator_niches": helper["brand_required_niches"],
            "preferred_creator_niches": helper["brand_preferred_niches"],
            "target_age_range": helper["brand_target_age"],
            "target_gender": helper["brand_target_gender"],
            "target_locations": helper["brand_target_locations"],
            "platforms": helper["brand_platforms"],
            "budget": helper["brand_budget"],
            "currency": helper["brand_currency"],
            "minimum_followers": helper["brand_min_followers"],
            "maximum_followers": helper["brand_max_followers"],
            "content_types": helper["brand_content_types"],
            "tone": helper["brand_tone"],
            "mandatory_requirements": helper["brand_mandatory"],
            "excluded_traits": helper["brand_excluded"],
        }
        
        creator = {
            "primary_niche": helper["creator_primary_niche"],
            "secondary_niches": helper["creator_secondary_niches"],
            "location": helper["creator_base_location"],
            "languages": creator_full.get("languages", ["English"]),
            "platforms": helper["creator_platforms"],
            "followers": helper["creator_followers"],
            "audience_age_range": helper["creator_audience_age"],
            "audience_gender_distribution": helper["creator_audience_gender"],
            "audience_locations": helper["creator_audience_locations"],
            "content_types": helper["creator_content_types"],
            "content_style": helper["creator_content_style"],
            "rate_card": helper["creator_rate_card"],
            "bio": helper["creator_bio"],
        }
        
        # Evaluate all 7 dimensions
        dimensions = {
            "niche_match": evaluate_niche_match(brand, creator),
            "audience_match": evaluate_audience_match(brand, creator),
            "geography_match": evaluate_geography_match(brand, creator),
            "platform_match": evaluate_platform_match(brand, creator),
            "budget_fit": evaluate_budget_fit(brand, creator),
            "creator_size_fit": evaluate_creator_size_fit(brand, creator),
            "campaign_fit": evaluate_campaign_fit(brand, creator),
        }
        
        label = determine_label(dimensions)
        reason = format_reason(dimensions, label)
        
        # Check if ambiguous (many partials, close to boundary)
        levels = [v.level for v in dimensions.values()]
        partial_count = sum(1 for v in levels if v in ["partial", "near", "uncertain"])
        weak_count = sum(1 for v in levels if v in ["weak", "mismatch", "outside"])
        if partial_count >= 3 or (weak_count == 2 and partial_count >= 2):
            ambiguous.append(f"{brand_id}-{creator_id}")
        
        label_counts[label] += 1
        
        annotated.append({
            "brand_id": brand_id,
            "creator_id": creator_id,
            "label": label,
            "reasoning": {k: v.level for k, v in dimensions.items()},
            "reason": reason
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Annotated {i + 1}/{len(pairs)} pairs...")
    
    # Save labels.json
    output = {
        "meta": data["meta"],
        "labels": annotated
    }
    with open("data/labels.json", "w") as f:
        json.dump(output, f, indent=2)
    
    # Generate report
    total = len(annotated)
    good_pct = label_counts["good"] / total * 100
    maybe_pct = label_counts["maybe"] / total * 100
    poor_pct = label_counts["poor"] / total * 100
    
    # Consistency checks
    pair_set = set()
    duplicates = []
    blank_labels = []
    invalid_labels = []
    missing_reasoning = []
    
    for item in annotated:
        key = (item["brand_id"], item["creator_id"])
        if key in pair_set:
            duplicates.append(key)
        pair_set.add(key)
        
        if not item["label"]:
            blank_labels.append(key)
        if item["label"] not in ["good", "maybe", "poor"]:
            invalid_labels.append((key, item["label"]))
        
        required_dims = ["niche_match", "audience_match", "geography_match", 
                        "platform_match", "budget_fit", "creator_size_fit", "campaign_fit"]
        for dim in required_dims:
            if dim not in item["reasoning"] or not item["reasoning"][dim]:
                missing_reasoning.append((key, dim))
    
    report = f"""# Annotation Report

## Summary
- **Total pairs**: {total}
- **Good**: {label_counts['good']} ({good_pct:.1f}%)
- **Maybe**: {label_counts['maybe']} ({maybe_pct:.1f}%)
- **Poor**: {label_counts['poor']} ({poor_pct:.1f}%)

## Consistency Checks
- **All 400 pairs present**: {'✓' if total == 400 else '✗'} ({total}/400)
- **No duplicate pairs**: {'✓' if not duplicates else '✗'} ({len(duplicates)} duplicates)
- **No blank labels**: {'✓' if not blank_labels else '✗'} ({len(blank_labels)} blank)
- **Only valid label values**: {'✓' if not invalid_labels else '✗'} ({len(invalid_labels)} invalid)
- **All 7 reasoning dimensions populated**: {'✓' if not missing_reasoning else '✗'} ({len(missing_reasoning)} missing)

## Ambiguous/Borderline Cases
- **Count**: {len(ambiguous)}
- **Pairs**: {', '.join(ambiguous[:20])}{'...' if len(ambiguous) > 20 else ''}

## Data/Annotation Issues Discovered
"""
    issues = []
    if duplicates:
        issues.append(f"Duplicate pairs found: {duplicates}")
    if blank_labels:
        issues.append(f"Blank labels: {blank_labels}")
    if invalid_labels:
        issues.append(f"Invalid labels: {invalid_labels}")
    if missing_reasoning:
        issues.append(f"Missing reasoning dimensions: {missing_reasoning}")
    if len(ambiguous) > total * 0.15:
        issues.append(f"High number of ambiguous cases ({len(ambiguous)}/{total})")
    
    if not issues:
        report += "None discovered.\n"
    else:
        for issue in issues:
            report += f"- {issue}\n"
    
    with open("evals/annotation_report.md", "w") as f:
        f.write(report)
    
    print(f"\nDone!")
    print(f"  Saved labels to data/labels.json")
    print(f"  Saved report to evals/annotation_report.md")
    print(f"  Distribution: good={label_counts['good']}, maybe={label_counts['maybe']}, poor={label_counts['poor']}")
    print(f"  Ambiguous cases: {len(ambiguous)}")

if __name__ == "__main__":
    main()