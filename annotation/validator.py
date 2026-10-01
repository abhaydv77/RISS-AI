"""Strict JSON parsing and pair-aware annotation validation."""
import json

from .schema import ANNOTATION_FIELDS, DIMENSIONS, LABELS


class AnnotationValidationError(ValueError):
    """A response is not a valid annotation for the requested pair."""


def parse_and_validate(raw, brand_id, creator_id):
    if isinstance(raw, dict):
        value = raw
    elif isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AnnotationValidationError(f"malformed JSON: {exc.msg} at line {exc.lineno} column {exc.colno}") from exc
    else:
        raise AnnotationValidationError("response must be a JSON string or object")
    if not isinstance(value, dict):
        raise AnnotationValidationError("top-level JSON value must be an object")
    missing = ANNOTATION_FIELDS - value.keys()
    extra = value.keys() - ANNOTATION_FIELDS
    if missing:
        raise AnnotationValidationError(f"missing required fields: {', '.join(sorted(missing))}")
    if extra:
        raise AnnotationValidationError(f"unexpected fields: {', '.join(sorted(extra))}")
    if value["brand_id"] != brand_id:
        raise AnnotationValidationError(f"brand_id mismatch: expected {brand_id!r}, received {value['brand_id']!r}")
    if value["creator_id"] != creator_id:
        raise AnnotationValidationError(f"creator_id mismatch: expected {creator_id!r}, received {value['creator_id']!r}")
    if not isinstance(value["label"], str) or value["label"] not in LABELS:
        raise AnnotationValidationError(f"invalid label {value['label']!r}; expected one of {sorted(LABELS)}")
    reasoning = value["reasoning"]
    if not isinstance(reasoning, dict):
        raise AnnotationValidationError("reasoning must be an object")
    missing_dims = DIMENSIONS.keys() - reasoning.keys()
    extra_dims = reasoning.keys() - DIMENSIONS.keys()
    if missing_dims:
        raise AnnotationValidationError(f"missing reasoning dimensions: {', '.join(sorted(missing_dims))}")
    if extra_dims:
        raise AnnotationValidationError(f"unexpected reasoning dimensions: {', '.join(sorted(extra_dims))}")
    for field, allowed in DIMENSIONS.items():
        if not isinstance(reasoning[field], str) or reasoning[field] not in allowed:
            raise AnnotationValidationError(f"invalid {field} value {reasoning[field]!r}; expected one of {sorted(allowed)}")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise AnnotationValidationError("reason must be a non-empty string")
    # Return a fresh object in stable schema order and discard no invalid content.
    return {
        "brand_id": value["brand_id"], "creator_id": value["creator_id"],
        "label": value["label"],
        "reasoning": {field: reasoning[field] for field in DIMENSIONS},
        "reason": value["reason"].strip(),
    }
