import json
import math
from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder

from src.model import load_model
from src.preprocessing import load_brands, load_creators
from src.ranker import rank_brand

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

LABEL_TO_GRADE = {"good": 2, "maybe": 1, "poor": 0}


def load_labels() -> list[dict[str, Any]]:
    with open(DATA_DIR / "labels.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["labels"]


def _precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = ranked_ids[:k]
    relevant_in_top_k = sum(1 for cid in top_k if cid in relevant_ids)
    denominator = min(k, len(ranked_ids))
    return relevant_in_top_k / denominator if denominator > 0 else 0.0


def _recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = ranked_ids[:k]
    relevant_in_top_k = sum(1 for cid in top_k if cid in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def _mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for i, cid in enumerate(ranked_ids):
        if cid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at_k(ranked_ids: list[str], graded_relevance: dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k]):
        grade = graded_relevance.get(cid, 0)
        dcg += (2**grade - 1) / math.log2(i + 2)

    ideal_grades = sorted(graded_relevance.values(), reverse=True)[:k]
    idcg = 0.0
    for i, grade in enumerate(ideal_grades):
        idcg += (2**grade - 1) / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_metrics(
    ranked_ids: list[str],
    labels: dict[str, str],
) -> dict[str, float]:
    relevant_ids = {cid for cid, label in labels.items() if label == "good"}
    graded_relevance = {cid: LABEL_TO_GRADE.get(label, 0) for cid, label in labels.items()}

    return {
        "precision_at_3": _precision_at_k(ranked_ids, relevant_ids, 3),
        "precision_at_5": _precision_at_k(ranked_ids, relevant_ids, 5),
        "recall_at_3": _recall_at_k(ranked_ids, relevant_ids, 3),
        "recall_at_5": _recall_at_k(ranked_ids, relevant_ids, 5),
        "mrr": _mrr(ranked_ids, relevant_ids),
        "ndcg_at_5": _ndcg_at_k(ranked_ids, graded_relevance, 5),
    }


def evaluate_brand(
    model: CrossEncoder,
    brand: dict[str, Any],
    creators: list[dict[str, Any]],
    brand_labels: list[dict[str, Any]],
) -> dict[str, Any]:
    results = rank_brand(model, brand, creators, k=len(creators))
    ranked_ids = [r["creator_id"] for r in results]

    labels_dict: dict[str, str] = {}
    graded_relevance: dict[str, int] = {}
    for item in brand_labels:
        labels_dict[item["creator_id"]] = item["label"]
        graded_relevance[item["creator_id"]] = LABEL_TO_GRADE.get(item["label"], 0)

    metrics = compute_metrics(ranked_ids, labels_dict)

    return {
        "brand_id": brand["brand_id"],
        "brand_name": brand.get("brand_name", ""),
        "metrics": metrics,
        "ranking": [
            {
                "creator_id": r["creator_id"],
                "score": r["score"],
                "label": labels_dict.get(r["creator_id"], "unknown"),
            }
            for r in results
        ],
    }


def run_evaluation() -> dict[str, Any]:
    brands = load_brands()
    creators = load_creators()
    all_labels = load_labels()

    labels_by_brand: dict[str, list[dict[str, Any]]] = {}
    for item in all_labels:
        labels_by_brand.setdefault(item["brand_id"], []).append(item)

    model = load_model()

    per_brand = []
    for brand in brands:
        brand_id = brand["brand_id"]
        brand_labels = labels_by_brand.get(brand_id, [])
        if not brand_labels:
            continue
        brand_result = evaluate_brand(model, brand, creators, brand_labels)
        per_brand.append(brand_result)

    overall_metrics: dict[str, float] = {}
    metric_names = [
        "precision_at_3",
        "precision_at_5",
        "recall_at_3",
        "recall_at_5",
        "mrr",
        "ndcg_at_5",
    ]
    for metric_name in metric_names:
        values = [b["metrics"][metric_name] for b in per_brand]
        overall_metrics[metric_name] = sum(values) / len(values) if values else 0.0

    report = {
        "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "overall": overall_metrics,
        "per_brand": per_brand,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "baseline_pretrained.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = run_evaluation()
    print(json.dumps(report["overall"], indent=2))
    print(f"\nReport saved to: {REPORTS_DIR / 'baseline_pretrained.json'}")
