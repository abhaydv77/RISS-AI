"""V0 rule and logistic baselines for brand-to-creator ranking.

This module defines training and evaluation commands but does not run them on
import. Example commands are documented in PHASE3_BASELINE_DESIGN.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

from feature_engineering import FEATURE_NAMES, engineer_features, feature_vector


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_REPORT = ROOT / "reports" / "phase3_baseline_v0.json"
LABELS = ("poor", "maybe", "good")
RELEVANCE = {"poor": 0, "maybe": 1, "good": 2}
SPLIT_KEYS = ("train_brand_ids", "validation_brand_ids", "test_brand_ids")
RANDOM_STATE = 42
LOGISTIC_C = 1.0


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_records() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Join labeled pairs to profiles; labels/reasoning never enter the features."""
    brands = _read_json(DATA_DIR / "brands.json")
    creators = _read_json(DATA_DIR / "creators.json")
    label_doc = _read_json(DATA_DIR / "labels.json")
    split = _read_json(DATA_DIR / "splits_v0.json")
    brand_by_id = {row["brand_id"]: row for row in brands}
    creator_by_id = {row["creator_id"]: row for row in creators}
    if len(brand_by_id) != len(brands) or len(creator_by_id) != len(creators):
        raise ValueError("Brand and creator IDs must be unique in profile files.")

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for label_row in label_doc["labels"]:
        key = (label_row["brand_id"], label_row["creator_id"])
        if key in seen:
            raise ValueError(f"Duplicate labeled pair: {key}")
        seen.add(key)
        if key[0] not in brand_by_id or key[1] not in creator_by_id:
            raise ValueError(f"Labeled pair references a missing profile: {key}")
        label = label_row["label"]
        if label not in RELEVANCE:
            raise ValueError(f"Unexpected label {label!r} for pair {key}")
        brand, creator = brand_by_id[key[0]], creator_by_id[key[1]]
        records.append({
            "brand_id": key[0],
            "creator_id": key[1],
            "brand_name": brand.get("brand_name", key[0]),
            "creator_name": creator.get("name", key[1]),
            "label": label,
            "features": engineer_features(brand, creator),
            "vector": feature_vector(brand, creator),
            "brand_profile": brand,
            "creator_profile": creator,
        })

    expected_pairs = {
        (brand_id, creator_id)
        for brand_id in brand_by_id
        for creator_id in creator_by_id
    }
    if seen != expected_pairs:
        missing_pairs = sorted(expected_pairs - seen)
        extra_pairs = sorted(seen - expected_pairs)
        raise ValueError(
            "This baseline expects a fully labeled brand-creator matrix; "
            f"missing={len(missing_pairs)}, extra={len(extra_pairs)}"
        )

    partitions: dict[str, str] = {}
    for key in SPLIT_KEYS:
        brand_ids = split.get(key, [])
        for brand_id in brand_ids:
            if brand_id in partitions:
                raise ValueError(f"Brand {brand_id} appears in multiple split partitions.")
            partitions[brand_id] = key
    if set(partitions) != set(brand_by_id):
        missing = sorted(set(brand_by_id) - set(partitions))
        unknown = sorted(set(partitions) - set(brand_by_id))
        raise ValueError(f"Split manifest must partition every brand; missing={missing}, unknown={unknown}")
    if {row["brand_id"] for row in records} != set(brand_by_id):
        raise ValueError("Labeled pairs must cover all brands in the split manifest.")
    return records, split, {"brands": brands, "creators": creators, "labels": label_doc}


def partition(records: Iterable[dict[str, Any]], brand_ids: Iterable[str]) -> list[dict[str, Any]]:
    allowed = set(brand_ids)
    return [row for row in records if row["brand_id"] in allowed]


def rule_score(record: dict[str, Any]) -> float:
    """Fixed, transparent rule baseline; no labels are consulted."""
    f = record["features"]
    brand, creator = record["brand_profile"], record["creator_profile"]

    niche = (
        0.65 * f["niche_primary_match"]
        + 0.20 * f["niche_preferred_match"]
        + 0.15 * min(f["niche_secondary_overlap"] / 2.0, 1.0)
    )
    audience = sum((
        f["audience_age_overlap"],
        f["audience_gender_match"],
        f["audience_location_overlap"],
    )) / 3.0
    geography = max(f["creator_in_target_country"], f["creator_in_mandatory_region"])
    # budget_to_min_rate_ratio is log1p(raw budget / minimum rate); cap it at a
    # raw ratio of 5 so unusually large budgets do not dominate the score.
    budget = (
        f["budget_fits_min"]
        + min(f["budget_to_min_rate_ratio"] / math.log1p(5.0), 1.0)
    ) / 2.0

    followers = max(0.0, float(creator.get("followers") or 0))
    min_followers = float(brand.get("minimum_followers") or 0)
    max_followers = float(brand.get("maximum_followers") or 0)
    if min_followers <= followers <= max_followers:
        size = 1.0
    elif min_followers * 0.8 <= followers <= max_followers * 1.2:
        size = 0.5
    else:
        size = 0.0

    components = (
        niche,
        audience,
        geography,
        f["platform_coverage"],
        budget,
        size,
        f["content_type_jaccard"],
        f["mandatory_requirements_met"],
    )
    return sum(components) / len(components) - 0.25 * f["has_excluded_trait"]


def expected_relevance_score(probabilities: dict[str, float]) -> float:
    return 2.0 * probabilities.get("good", 0.0) + probabilities.get("maybe", 0.0)


def _dcg(labels: list[str], k: int) -> float:
    total = 0.0
    for index, label in enumerate(labels[:k]):
        relevance = RELEVANCE[label]
        gain = (2**relevance) - 1
        total += gain / math.log2(index + 2)
    return total


def _ranked(records: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    # Python's stable sort preserves source order for equal scores; creator IDs
    # are not used in scoring or as tie-breaker features.
    return sorted(records, key=lambda row: row[score_key], reverse=True)


def ranking_metrics(records: list[dict[str, Any]], ks: tuple[int, ...] = (3, 5)) -> dict[str, Any]:
    """Compute per-brand top-K retrieval and graded ranking measures."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row["brand_id"]].append(row)

    per_brand: dict[str, dict[str, float | None]] = {}
    for brand_id, rows in sorted(groups.items()):
        ranked = _ranked(rows, "score")
        labels = [row["label"] for row in ranked]
        total_good = labels.count("good")
        metrics: dict[str, float | None] = {}
        for k in ks:
            top_labels = labels[:k]
            denom = min(k, len(labels))
            good_in_top = sum(label == "good" for label in top_labels)
            metrics[f"precision@{k}_good"] = good_in_top / denom if denom else None
            metrics[f"recall@{k}_good"] = good_in_top / total_good if total_good else None

        first_good_rank = next((index + 1 for index, label in enumerate(labels) if label == "good"), None)
        metrics["mrr_good"] = 1.0 / first_good_rank if first_good_rank is not None else 0.0
        max_k = max(ks)
        ideal_labels = sorted(labels, key=lambda label: RELEVANCE[label], reverse=True)
        ideal_dcg = _dcg(ideal_labels, max_k)
        metrics[f"ndcg@{max_k}"] = _dcg(labels, max_k) / ideal_dcg if ideal_dcg else 0.0
        per_brand[brand_id] = metrics

    macro: dict[str, float | None] = {}
    metric_names = next(iter(per_brand.values())).keys() if per_brand else ()
    for metric_name in metric_names:
        values = [row[metric_name] for row in per_brand.values() if row[metric_name] is not None]
        macro[metric_name] = sum(values) / len(values) if values else None
    return {"macro_brand_average": macro, "per_brand": per_brand}


def classification_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Secondary classification diagnostics aggregated over input brand groups."""
    matrix = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    for row in records:
        matrix[row["label"]][row["predicted_label"]] += 1
    total = sum(sum(cells.values()) for cells in matrix.values())
    accuracy = sum(matrix[label][label] for label in LABELS) / total if total else None
    class_f1: dict[str, float] = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in LABELS if actual != label)
        fn = sum(matrix[label][predicted] for predicted in LABELS if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        class_f1[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": sum(class_f1.values()) / len(LABELS),
        "per_class_f1": class_f1,
        "confusion_matrix_true_rows_predicted_columns": matrix,
    }


def _make_predictions(records: list[dict[str, Any]], model: Any | None, score_key: str) -> list[dict[str, Any]]:
    if model is None:
        predictions = [rule_score(row) for row in records]
        for row, score in zip(records, predictions):
            row[score_key] = float(score)
            row["predicted_label"] = None
        return records

    matrix = [row["vector"] for row in records]
    probability_rows = model.predict_proba(matrix)
    model_classes = list(model.classes_)
    for row, probability_row in zip(records, probability_rows):
        probabilities = {label: float(value) for label, value in zip(model_classes, probability_row)}
        row["probabilities"] = {label: probabilities.get(label, 0.0) for label in LABELS}
        row[score_key] = expected_relevance_score(row["probabilities"])
        row["predicted_label"] = model_classes[int(probability_row.argmax())]
    return records


def _fit_model(records: list[dict[str, Any]], class_weight: str | None) -> Any:
    # Lazy imports let the rule baseline and metric utilities remain importable
    # even before the optional ML dependency is installed.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = [row["vector"] for row in records]
    y = [row["label"] for row in records]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=LOGISTIC_C,
            class_weight=class_weight,
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
    )
    model.fit(x, y)
    return model


def _evaluate_model_variant(
    training_records: list[dict[str, Any]],
    evaluation_records: list[dict[str, Any]],
    class_weight: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = _fit_model(training_records, class_weight)
    # Clone no mutable profile/feature data into report rows.
    rows = [dict(row) for row in evaluation_records]
    _make_predictions(rows, model, "score")
    result = {
        "model": "L2 multinomial logistic regression",
        "class_weight": class_weight,
        "C": LOGISTIC_C,
        "random_state": RANDOM_STATE,
        "ranking_metrics": ranking_metrics(rows),
        "classification_metrics": classification_metrics(rows),
    }
    return result, rows


def _evaluate_rules(evaluation_records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [dict(row) for row in evaluation_records]
    _make_predictions(rows, None, "score")
    return {"ranking_metrics": ranking_metrics(rows)}, rows


def _leave_one_brand_out(records: list[dict[str, Any]], class_weight: str | None) -> dict[str, Any]:
    """Out-of-fold ranking estimates across development brands only."""
    brand_ids = sorted({row["brand_id"] for row in records})
    out_of_fold: list[dict[str, Any]] = []
    for held_out_brand in brand_ids:
        fold_train = [row for row in records if row["brand_id"] != held_out_brand]
        fold_eval = [row for row in records if row["brand_id"] == held_out_brand]
        model = _fit_model(fold_train, class_weight)
        predictions = [dict(row) for row in fold_eval]
        _make_predictions(predictions, model, "score")
        out_of_fold.extend(predictions)
    return {
        "held_out_brands": brand_ids,
        "ranking_metrics": ranking_metrics(out_of_fold),
        "classification_metrics": classification_metrics(out_of_fold),
    }


def _label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["label"] for row in records)
    return {label: counts.get(label, 0) for label in LABELS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _prediction_export(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["brand_id"]].append(row)
    exported = []
    for brand_id in sorted(grouped):
        for rank, row in enumerate(_ranked(grouped[brand_id], "score"), start=1):
            item = {
                "brand_id": brand_id,
                "creator_id": row["creator_id"],
                "rank": rank,
                "true_label": row["label"],
                "predicted_label": row.get("predicted_label"),
                "ranking_score": float(row["score"]),
            }
            if row.get("probabilities") is not None:
                item["probabilities"] = row["probabilities"]
            exported.append(item)
    return exported


def run_experiment(stage: str = "validation", report_path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    """Train/evaluate the requested stage and persist a reproducibility report.

    ``validation`` uses train brands to compare fixed model variants on the
    validation brand and does not score the test brand. ``final`` evaluates the
    validation stage, refits on train+validation, then scores the held-out test.
    """
    if stage not in {"validation", "final"}:
        raise ValueError("stage must be 'validation' or 'final'")
    records, split, source_data = load_records()
    train = partition(records, split["train_brand_ids"])
    validation = partition(records, split["validation_brand_ids"])
    test = partition(records, split["test_brand_ids"])

    # The deterministic rules are evaluated on validation in both stages and
    # on test only when final evaluation is explicitly requested.
    rule_val, rule_val_rows = _evaluate_rules(validation)
    model_validation: dict[str, Any] = {}
    model_val_rows: dict[str, list[dict[str, Any]]] = {}
    variants = ((None, "unweighted"), ("balanced", "balanced"))
    for class_weight, model_name in variants:
        result, rows = _evaluate_model_variant(train, validation, class_weight)
        model_validation[model_name] = result
        model_val_rows[model_name] = rows

    report: dict[str, Any] = {
        "experiment": "riss-ai-phase3-ranking-v0",
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_summary": {
            "brands": len(source_data["brands"]),
            "creators": len(source_data["creators"]),
            "labeled_pairs": len(records),
            "label_counts": _label_counts(records),
            "feature_count": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
        },
        "split_manifest": split,
        "partition_label_counts": {
            "train": _label_counts(train),
            "validation": _label_counts(validation),
            "test": _label_counts(test),
        },
        "evaluation_contract": {
            "good_relevance_for_precision_recall_mrr": True,
            "graded_relevance": RELEVANCE,
            "ndcg_gain": "2^relevance - 1",
            "metrics": ["precision@3_good", "precision@5_good", "recall@3_good", "recall@5_good", "mrr_good", "ndcg@5"],
            "averaging": "macro average over brand groups",
        },
        "rule_baseline": {"validation": rule_val},
        "logistic_models": {
            "leave_one_train_brand_out": {
                "unweighted": _leave_one_brand_out(train, None),
                "balanced": _leave_one_brand_out(train, "balanced"),
            },
            "validation": model_validation,
        },
        "validation_predictions": {
            "rule_baseline": _prediction_export(rule_val_rows),
            **{name: _prediction_export(rows) for name, rows in model_val_rows.items()},
        },
        "configuration": {
            "random_state": RANDOM_STATE,
            "logistic_c": LOGISTIC_C,
            "scaling": "StandardScaler fitted on training data only",
            "feature_ids_used": False,
            "labels_or_reasoning_used_as_features": False,
            "training_pairs": len(train),
            "validation_pairs": len(validation),
            "test_pairs": len(test),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "scikit_learn": _package_version("scikit-learn"),
            "git_revision": _git_revision(),
            "input_sha256": {
                name: _sha256(DATA_DIR / filename)
                for name, filename in {
                    "brands": "brands.json",
                    "creators": "creators.json",
                    "labels": "labels.json",
                    "split_manifest": "splits_v0.json",
                }.items()
            },
            "code_sha256": {
                "feature_engineering.py": _sha256(ROOT / "feature_engineering.py"),
                "ranking_experiment.py": _sha256(ROOT / "ranking_experiment.py"),
            },
        },
    }

    if stage == "final":
        # Only after model selection on validation: refit on train+validation,
        # then use the test brand once for the reported final result.
        final_training = train + validation
        rule_test, rule_test_rows = _evaluate_rules(test)
        final_results: dict[str, Any] = {}
        final_prediction_rows: dict[str, list[dict[str, Any]]] = {}
        for class_weight, model_name in variants:
            result, rows = _evaluate_model_variant(final_training, test, class_weight)
            final_results[model_name] = result
            final_prediction_rows[model_name] = rows
        report["rule_baseline"]["test"] = rule_test
        report["logistic_models"]["test_after_refit_on_train_plus_validation"] = final_results
        report["test_predictions"] = {
            "rule_baseline": _prediction_export(rule_test_rows),
            **{name: _prediction_export(rows) for name, rows in final_prediction_rows.items()},
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validation", "final"), default="validation")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_experiment(stage=args.stage, report_path=args.report)
    print(json.dumps({
        "stage": report["stage"],
        "report": str(args.report),
        "validation_rule_metrics": report["rule_baseline"]["validation"]["ranking_metrics"]["macro_brand_average"],
        "validation_model_metrics": {
            name: result["ranking_metrics"]["macro_brand_average"]
            for name, result in report["logistic_models"]["validation"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
