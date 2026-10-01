"""Prompt and compact profile context for one brand–creator pair."""
import json


BRAND_FIELDS = (
    "brand_id", "brand_name", "industry", "product", "campaign_title", "campaign_goal",
    "campaign_description", "target_audience", "target_age_range", "target_gender",
    "target_locations", "required_creator_niches", "preferred_creator_niches",
    "creator_size_preference", "minimum_followers", "maximum_followers", "budget", "currency",
    "content_types", "platforms", "tone", "mandatory_requirements", "preferred_traits", "excluded_traits",
)
CREATOR_FIELDS = (
    "creator_id", "name", "primary_niche", "secondary_niches", "bio", "location", "languages",
    "platforms", "followers", "engagement_rate", "average_views", "audience_age_range",
    "audience_gender_distribution", "audience_locations", "content_types", "content_style",
    "rate_card", "rate_currency", "past_brand_categories", "interests",
)


def build_pair_context(brand, creator):
    return {
        "brand": {key: brand[key] for key in BRAND_FIELDS if key in brand},
        "creator": {key: creator[key] for key in CREATOR_FIELDS if key in creator},
    }


GUIDE_RULES = """Use these annotation guide definitions:
- good: strong fit across most dimensions; mandatory requirements, audience, niche/content, budget/size, geography/platform generally align.
- maybe: partial fit with notable gaps (typically 2+ important gaps or one major blocker); when in doubt, choose maybe.
- poor: weak fit or clear mismatch, such as fundamental audience/niche/geography mismatch, far outside size/budget, no platform overlap, excluded traits, or unmet mandatory requirements.

Reasoning dimensions:
- niche_match: strong when primary niche is required; partial when primary is preferred or secondary overlaps required/preferred; weak when none overlap.
- audience_match: strong when age significantly overlaps, target gender matches (all always matches; female/male requires at least 60%), and audience location overlaps; partial for 2 of those 3; weak for 0–1.
- geography_match: strong when creator base country is in target locations or explicitly required; partial when audience locations overlap but creator is based elsewhere; weak when both are outside.
- platform_match: strong when all brand platforms are among creator platforms; partial for any overlap; weak for none.
- budget_fit: fit when brand budget is at least the creator rate-card minimum; uncertain when budget is below the minimum but at least half of it, or when the rate card is ambiguous; mismatch when budget is less than half the minimum. Compare currencies only when the supplied currency information permits it.
- creator_size_fit: fit within the follower range; near within 20% of a boundary; outside when farther away.
- campaign_fit: strong when content overlaps strongly, style matches tone, no excluded traits apply, and mandatory requirements are met; partial for some overlap/style or minor conflict; weak for mismatch, excluded traits, or unmet mandatory requirements.

Mandatory requirements are hard filters (at best maybe when unmet, usually poor). Excluded traits are strong negative signals. For location-specific campaigns, an out-of-target creator base is weak geography unless the audience overlaps, which is partial. Use only provided facts; do not assume or invent missing information. If information needed for a judgment is missing or ambiguous, reflect that in the relevant allowed category and concise reason."""


def build_prompt(context):
    return (
        "Annotate exactly one brand–creator pair using the supplied profiles. Follow the annotation guide rules below. "
        "Evaluate all seven reasoning dimensions and choose one overall label. Give a concise factual reason. "
        "Use only the supplied brand and creator information; never invent missing facts. Return ONLY valid JSON matching "
        "the required schema, with no markdown or surrounding text.\n\n"
        "Required JSON schema: {\"brand_id\": string, \"creator_id\": string, \"label\": \"good|maybe|poor\", "
        "\"reasoning\": {\"niche_match\": \"strong|partial|weak\", \"audience_match\": \"strong|partial|weak\", "
        "\"geography_match\": \"strong|partial|weak\", \"platform_match\": \"strong|partial|weak\", "
        "\"budget_fit\": \"fit|uncertain|mismatch\", \"creator_size_fit\": \"fit|near|outside\", "
        "\"campaign_fit\": \"strong|partial|weak\"}, \"reason\": string}.\n\n"
        f"ANNOTATION GUIDE RULES\n{GUIDE_RULES}\n\nPAIR CONTEXT (JSON)\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )
