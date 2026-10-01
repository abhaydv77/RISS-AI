"""Annotation schema and fixed terminology from reports/ANNOTATION_GUIDE.md."""

LABELS = frozenset({"good", "maybe", "poor"})
DIMENSIONS = {
    "niche_match": frozenset({"strong", "partial", "weak"}),
    "audience_match": frozenset({"strong", "partial", "weak"}),
    "geography_match": frozenset({"strong", "partial", "weak"}),
    "platform_match": frozenset({"strong", "partial", "weak"}),
    "budget_fit": frozenset({"fit", "uncertain", "mismatch"}),
    "creator_size_fit": frozenset({"fit", "near", "outside"}),
    "campaign_fit": frozenset({"strong", "partial", "weak"}),
}
ANNOTATION_FIELDS = frozenset({"brand_id", "creator_id", "label", "reasoning", "reason"})

