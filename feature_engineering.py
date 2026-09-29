"""Explainable V0 features for brand–creator ranking.

The public entry point is :func:`engineer_features`; it accepts the raw profile
dictionaries used in ``data/brands.json`` and ``data/creators.json``.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


# Units of local currency per USD. Fixed deliberately so training and serving
# use identical conversions; update only as an explicit model-version change.
FX_PER_USD = {"USD": 1.0, "INR": 83.0, "KRW": 1300.0, "JPY": 150.0,
              "EUR": 0.92, "SEK": 10.5}
_SYMBOL_CURRENCY = {"₹": "INR", "₩": "KRW", "¥": "JPY", "$": "USD",
                    "€": "EUR", "kr": "SEK"}
_COUNTRY_ALIASES = {
    "us": "united states", "usa": "united states", "united states of america": "united states",
    "uk": "united kingdom", "south korea": "south korea", "korea": "south korea",
    "uae": "united arab emirates", "republic of korea": "south korea",
}
_REGIONS = {
    "south asia": {"india", "pakistan", "bangladesh", "sri lanka", "nepal", "bhutan", "maldives", "afghanistan"},
    "east asia": {"china", "japan", "south korea", "north korea", "mongolia", "taiwan"},
}


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _country(value: Any) -> str:
    """Extract a country from a ``City, Country`` value and normalize aliases."""
    text = _norm(value)
    if not text:
        return ""
    country = text.rsplit(",", 1)[-1].strip()
    return _COUNTRY_ALIASES.get(country, country)


def parse_age_range(value: Any) -> tuple[float, float] | None:
    """Parse ranges such as ``18-30`` or ``18–30``; return None if unavailable."""
    matches = re.findall(r"\d+(?:\.\d+)?", str(value or ""))
    if len(matches) < 2:
        return None
    low, high = float(matches[0]), float(matches[1])
    return (low, high) if low <= high else (high, low)


def _amounts(value: Any) -> list[float]:
    # Commas, decimal points, and a possible sign; currency symbols are ignored.
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", str(value or ""))]


def parse_rate_card(value: Any, currency: str | None = None) -> tuple[float, float] | None:
    """Return min/max rate in USD. Currency symbols override supplied currency."""
    text = str(value or "")
    numbers = _amounts(text)
    if not numbers:
        return None
    cur = (currency or "INR").upper()
    for symbol, detected in _SYMBOL_CURRENCY.items():
        if symbol in text:
            cur = detected
            break
    factor = FX_PER_USD.get(cur)
    if factor is None:
        return None
    lo, hi = (numbers[0], numbers[1]) if len(numbers) > 1 else (numbers[0], numbers[0])
    return lo / factor, hi / factor


def _as_set(value: Any) -> set[str]:
    if isinstance(value, str):
        value = [value]
    return {_norm(x) for x in (value or []) if _norm(x)}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left | right else 0.0


def _age_overlap(brand: Any, creator: Any) -> float:
    b, c = parse_age_range(brand), parse_age_range(creator)
    if not b or not c:
        return 0.0
    span = b[1] - b[0]
    if span <= 0:
        return float(c[0] <= b[0] <= c[1])
    return max(0.0, min(b[1], c[1]) - max(b[0], c[0])) / span


def _region_match(country: str, target: str) -> bool:
    target = _COUNTRY_ALIASES.get(_norm(target), _norm(target))
    if target == country:
        return True
    return country in _REGIONS.get(target, set())


def _mandatory_location(brand: Mapping[str, Any]) -> str | None:
    for requirement in brand.get("mandatory_requirements", []) or []:
        match = re.search(r"must be based in\s+(.+)", str(requirement), re.I)
        if match:
            return match.group(1).strip().rstrip(".")
    return None


def _trait_evidence(trait: str, creator: Mapping[str, Any]) -> bool:
    """Conservative deterministic evidence matching for the V0 exclusion flag."""
    haystack = " ".join([
        str(creator.get("bio", "")),
        " ".join(map(str, creator.get("content_types", []) or [])),
        str(creator.get("primary_niche", "")),
        " ".join(map(str, creator.get("secondary_niches", []) or [])),
    ]).lower()
    trait = _norm(trait)
    if not trait or "audience" in trait and any(x in trait for x in ("male", "female")):
        return False  # audience traits are handled from the observed distribution
    # Exact phrase is reliable; otherwise require meaningful non-stopword terms.
    if trait in haystack:
        return True
    words = [w for w in re.findall(r"[a-z]+", trait) if w not in {"only", "focused", "focus", "content", "audience"}]
    return len(words) >= 2 and all(re.search(r"\b" + re.escape(w) + r"\b", haystack) for w in words)


FEATURE_NAMES = (
    "niche_primary_match", "niche_secondary_overlap", "niche_preferred_match",
    "audience_age_overlap", "audience_gender_match", "audience_location_overlap",
    "creator_in_target_country", "creator_in_mandatory_region", "platform_coverage",
    "platform_all_covered", "budget_to_min_rate_ratio", "budget_fits_min",
    "followers_percentile_in_range", "followers_log", "creator_size_tier",
    "content_type_jaccard", "has_excluded_trait", "mandatory_requirements_met",
    "engagement_rate", "avg_views_per_follower",
)


def engineer_features(brand: Mapping[str, Any], creator: Mapping[str, Any]) -> dict[str, float]:
    """Compute the 20 deterministic V0 features from raw profile dictionaries.

    ``mandatory_requirements_met`` is a fraction over requirements that can be
    checked from the supplied fields (location, language, content, platform,
    gender, camera). Unrecognized requirements are omitted from its denominator.
    """
    required = _as_set(brand.get("required_creator_niches"))
    preferred = _as_set(brand.get("preferred_creator_niches"))
    primary = _norm(creator.get("primary_niche"))
    creator_niches = _as_set(creator.get("secondary_niches")) | ({primary} if primary else set())
    target_countries = {_COUNTRY_ALIASES.get(_norm(x), _norm(x)) for x in brand.get("target_locations", []) or []}
    audience_countries = {_COUNTRY_ALIASES.get(_norm(x), _norm(x)) for x in creator.get("audience_locations", []) or []}
    location = _country(creator.get("location"))
    platforms, creator_platforms = _as_set(brand.get("platforms")), _as_set(creator.get("platforms"))
    platform_coverage = len(platforms & creator_platforms) / len(platforms) if platforms else 0.0
    budget = float(brand.get("budget") or 0)
    budget_currency = str(brand.get("currency") or "USD").upper()
    budget_usd = budget / FX_PER_USD.get(budget_currency, 1.0)
    rates = parse_rate_card(creator.get("rate_card"), creator.get("rate_currency") or "INR")
    min_rate = rates[0] if rates else 0.0
    raw_ratio = budget_usd / min_rate if min_rate > 0 else 0.0
    min_followers = float(brand.get("minimum_followers") or 0)
    max_followers = float(brand.get("maximum_followers") or 0)
    followers = max(0.0, float(creator.get("followers") or 0))
    percentile = ((followers - min_followers) / (max_followers - min_followers)
                  if max_followers > min_followers else float(followers >= min_followers))
    gender = { _norm(k): float(v or 0) for k, v in (creator.get("audience_gender_distribution") or {}).items() }
    target_gender = _norm(brand.get("target_gender", "all"))
    gender_match = target_gender == "all" or gender.get(target_gender, 0.0) >= 0.60
    content = _as_set(brand.get("content_types"))
    creator_content = _as_set(creator.get("content_types"))

    excluded_traits = _as_set(brand.get("excluded_traits"))
    excluded = any(_trait_evidence(t, creator) for t in excluded_traits)
    for trait in excluded_traits:
        if "audience" in trait or "male-only" in trait or "female-only" in trait:
            if re.search(r"\bmale\b", trait):
                excluded |= gender.get("male", 0.0) >= 0.60
            if re.search(r"\bfemale\b", trait):
                excluded |= gender.get("female", 0.0) >= 0.60

    known, met = 0, 0
    base_req = _mandatory_location(brand)
    if base_req:
        known += 1
        req_countries = [_norm(v) for v in re.split(r"\s+or\s+|,", base_req, flags=re.I)]
        met += int(any(_region_match(location, v) for v in req_countries))
    langs = _as_set(creator.get("languages"))
    for req in brand.get("mandatory_requirements", []) or []:
        req = _norm(req)
        if "must be based in" in req:
            continue
        if "must post in" in req:
            values = [x for x in re.split(r"\s+or\s+|,", req.split("must post in", 1)[1]) if x.strip()]
            known += 1; met += int(bool(langs & _as_set(values)))
        elif "must have" in req and "content" in req:
            phrase = req.split("must have", 1)[1].replace("content", "").strip()
            requested = [x.strip() for x in phrase.split(" and ") if x.strip()]
            available = creator_content | creator_niches
            known += 1; met += int(bool(requested) and all(x in available for x in requested))
        elif "must have" in req and "audience" in req and any(g in req for g in ("male", "female")):
            wanted = "male" if re.search(r"\bmale\b", req) else "female"
            known += 1; met += int(gender.get(wanted, 0.0) >= 0.60)
        elif "must create" in req:
            phrase = req.split("must create", 1)[1].replace("content", "").strip()
            if "video" in phrase:
                # Video capability is inferred from YouTube/TikTok presence.
                known += 1; met += int(bool(creator_platforms & {"youtube", "tiktok"}))
            else:
                known += 1; met += int(_norm(phrase) in creator_platforms)
        elif "must be" in req and any(g in req for g in ("male", "female")) and "audience" in req:
            wanted = "male" if "male" in req else "female"
            known += 1; met += int(gender.get(wanted, 0.0) >= 0.60)
        elif "comfortable on camera" in req:
            # No explicit camera-comfort field; leave this requirement unscored.
            pass

    features = {
        "niche_primary_match": float(primary in required),
        "niche_secondary_overlap": float(len(_as_set(creator.get("secondary_niches")) & (required | preferred))),
        "niche_preferred_match": float(primary in preferred),
        "audience_age_overlap": _age_overlap(brand.get("target_age_range"), creator.get("audience_age_range")),
        "audience_gender_match": float(gender_match),
        "audience_location_overlap": _jaccard(target_countries, audience_countries),
        "creator_in_target_country": float(any(_region_match(location, t) for t in target_countries)),
        "creator_in_mandatory_region": float(bool(base_req) and any(_region_match(location, v) for v in re.split(r"\s+or\s+|,", base_req, flags=re.I))),
        "platform_coverage": platform_coverage,
        "platform_all_covered": float(bool(platforms) and platforms <= creator_platforms),
        # log1p keeps the requested ratio feature stable for very large ratios.
        "budget_to_min_rate_ratio": math.log1p(raw_ratio),
        "budget_fits_min": float(bool(rates) and budget_usd >= min_rate),
        "followers_percentile_in_range": min(1.0, max(0.0, percentile)),
        "followers_log": math.log10(followers + 1),
        "creator_size_tier": float(0 if followers < 50_000 else 1 if followers < 500_000 else 2 if followers < 2_000_000 else 3),
        "content_type_jaccard": _jaccard(content, creator_content),
        "has_excluded_trait": float(excluded),
        "mandatory_requirements_met": met / known if known else 1.0,
        "engagement_rate": max(0.0, float(creator.get("engagement_rate") or 0)),
        "avg_views_per_follower": max(0.0, float(creator.get("average_views") or 0)) / followers if followers else 0.0,
    }
    return features


def feature_vector(brand: Mapping[str, Any], creator: Mapping[str, Any]) -> list[float]:
    """Return V0 features in the stable order given by :data:`FEATURE_NAMES`."""
    features = engineer_features(brand, creator)
    return [features[name] for name in FEATURE_NAMES]
