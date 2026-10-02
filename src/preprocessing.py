import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_brands() -> list[dict[str, Any]]:
    with open(DATA_DIR / "brands.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_creators() -> list[dict[str, Any]]:
    with open(DATA_DIR / "creators.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _join(items: list[str] | None) -> str:
    if not items:
        return ""
    return ", ".join(str(item) for item in items)


def brand_to_query(brand: dict[str, Any]) -> str:
    parts: list[str] = []

    name = brand.get("brand_name", "")
    if name:
        parts.append(f"Brand: {name}")

    industry = brand.get("industry", "")
    if industry:
        parts.append(f"Industry: {industry}")

    product = brand.get("product", "")
    if product:
        parts.append(f"Product: {product}")

    campaign_title = brand.get("campaign_title", "")
    if campaign_title:
        parts.append(f"Campaign: {campaign_title}")

    goal = brand.get("campaign_goal", "")
    if goal:
        parts.append(f"Goal: {goal}")

    description = brand.get("campaign_description", "")
    if description:
        parts.append(f"Description: {description}")

    audience = brand.get("target_audience", "")
    if audience:
        parts.append(f"Target audience: {audience}")

    age_range = brand.get("target_age_range", "")
    if age_range:
        parts.append(f"Target age: {age_range}")

    gender = brand.get("target_gender", "")
    if gender:
        parts.append(f"Target gender: {gender}")

    locations = _join(brand.get("target_locations"))
    if locations:
        parts.append(f"Target locations: {locations}")

    required_niches = _join(brand.get("required_creator_niches"))
    if required_niches:
        parts.append(f"Required niches: {required_niches}")

    preferred_niches = _join(brand.get("preferred_creator_niches"))
    if preferred_niches:
        parts.append(f"Preferred niches: {preferred_niches}")

    size_pref = brand.get("creator_size_preference", "")
    if size_pref:
        parts.append(f"Creator size: {size_pref}")

    min_followers = brand.get("minimum_followers")
    max_followers = brand.get("maximum_followers")
    if min_followers is not None and max_followers is not None:
        parts.append(f"Follower range: {min_followers}-{max_followers}")

    content_types = _join(brand.get("content_types"))
    if content_types:
        parts.append(f"Content types: {content_types}")

    platforms = _join(brand.get("platforms"))
    if platforms:
        parts.append(f"Platforms: {platforms}")

    tone = brand.get("tone", "")
    if tone:
        parts.append(f"Tone: {tone}")

    mandatory = _join(brand.get("mandatory_requirements"))
    if mandatory:
        parts.append(f"Mandatory requirements: {mandatory}")

    preferred_traits = _join(brand.get("preferred_traits"))
    if preferred_traits:
        parts.append(f"Preferred traits: {preferred_traits}")

    excluded_traits = _join(brand.get("excluded_traits"))
    if excluded_traits:
        parts.append(f"Excluded traits: {excluded_traits}")

    return ". ".join(parts)


def creator_to_passage(creator: dict[str, Any]) -> str:
    parts: list[str] = []

    name = creator.get("name", "")
    if name:
        parts.append(f"Creator: {name}")

    niche = creator.get("primary_niche", "")
    if niche:
        parts.append(f"Primary niche: {niche}")

    secondary = _join(creator.get("secondary_niches"))
    if secondary:
        parts.append(f"Secondary niches: {secondary}")

    bio = creator.get("bio", "")
    if bio:
        parts.append(f"Bio: {bio}")

    location = creator.get("location", "")
    if location:
        parts.append(f"Location: {location}")

    languages = _join(creator.get("languages"))
    if languages:
        parts.append(f"Languages: {languages}")

    platforms = _join(creator.get("platforms"))
    if platforms:
        parts.append(f"Platforms: {platforms}")

    followers = creator.get("followers")
    if followers is not None:
        parts.append(f"Followers: {followers}")

    engagement = creator.get("engagement_rate")
    if engagement is not None:
        parts.append(f"Engagement rate: {engagement}")

    avg_views = creator.get("average_views")
    if avg_views is not None:
        parts.append(f"Average views: {avg_views}")

    audience_age = creator.get("audience_age_range", "")
    if audience_age:
        parts.append(f"Audience age: {audience_age}")

    gender_dist = creator.get("audience_gender_distribution", {})
    if gender_dist:
        dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(gender_dist.items()))
        parts.append(f"Audience gender: {dist_str}")

    audience_locations = _join(creator.get("audience_locations"))
    if audience_locations:
        parts.append(f"Audience locations: {audience_locations}")

    content_types = _join(creator.get("content_types"))
    if content_types:
        parts.append(f"Content types: {content_types}")

    style = creator.get("content_style", "")
    if style:
        parts.append(f"Content style: {style}")

    rate_card = creator.get("rate_card", "")
    if rate_card:
        parts.append(f"Rate card: {rate_card}")

    past_brands = _join(creator.get("past_brand_categories"))
    if past_brands:
        parts.append(f"Past brand categories: {past_brands}")

    interests = _join(creator.get("interests"))
    if interests:
        parts.append(f"Interests: {interests}")

    posting_freq = creator.get("posting_frequency", "")
    if posting_freq:
        parts.append(f"Posting frequency: {posting_freq}")

    return ". ".join(parts)


def make_pairs(
    brand: dict[str, Any], creators: list[dict[str, Any]]
) -> list[dict[str, str]]:
    query = brand_to_query(brand)
    brand_id = brand.get("brand_id", "")
    pairs: list[dict[str, str]] = []
    for creator in creators:
        pairs.append(
            {
                "query": query,
                "passage": creator_to_passage(creator),
                "brand_id": brand_id,
                "creator_id": creator.get("creator_id", ""),
            }
        )
    return pairs
