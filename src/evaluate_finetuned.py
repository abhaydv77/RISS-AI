import json
from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder

try:
    from src.evaluation import compute_metrics, load_labels
    from src.preprocessing import load_brands, load_creators
    from src.ranker import rank_brand
except ImportError:
    from evaluation import compute_metrics, load_labels
    from preprocessing import load_brands, load_creators
    from ranker import rank_brand

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

TEST_BRANDS = ["b01", "b02"]


def load_finetuned_model() -> tuple[CrossEncoder, str]:
    artifact_dirs = sorted(ARTIFACTS_DIR.glob("fine_tuned_*"))
    if not artifact_dirs:
        raise FileNotFoundError("No fine-tuned model found in artifacts/")
    latest = artifact_dirs[-1]
    model = CrossEncoder(str(latest))
    return model, str(latest)


def evaluate_on_test(
    model: CrossEncoder,
    test_brand_ids: list[str],
) -> dict[str, Any]:
    brands = load_brands()
    creators = load_creators()
    all_labels = load_labels()

    labels_by_brand: dict[str, dict[str, str]] = {}
    for item in all_labels:
        labels_by_brand.setdefault(item["brand_id"], {})[item["creator_id"]] = item["label"]

    test_brands = [b for b in brands if b["brand_id"] in test_brand_ids]

    per_brand = []
    all_ranked_ids = []
    all_labels_dict = {}

    for brand in test_brands:
        brand_id = brand["brand_id"]
        brand_labels = labels_by_brand.get(brand_id, {})

        results = rank_brand(model, brand, creators, k=len(creators))
        ranked_ids = [r["creator_id"] for r in results]

        all_ranked_ids.extend(ranked_ids)
        all_labels_dict.update(brand_labels)

        metrics = compute_metrics(ranked_ids, brand_labels)
        per_brand.append(
            {
                "brand_id": brand_id,
                "brand_name": brand.get("brand_name", ""),
                "metrics": metrics,
                "ranking": [
                    {
                        "creator_id": r["creator_id"],
                        "score": r["score"],
                        "label": brand_labels.get(r["creator_id"], "unknown"),
                    }
                    for r in results
                ],
            }
        )

    overall = compute_metrics(all_ranked_ids, all_labels_dict)

    return {
        "test_brands": test_brand_ids,
        "num_test_pairs": len(all_labels_dict),
        "overall": overall,
        "per_brand": per_brand,
    }


def load_baseline_report() -> dict[str, Any]:
    with open(DATA_DIR / "baseline_pretrained.json", "r", encoding="utf-8") as f:
        return json.load(f)


def create_comparison(
    baseline: dict[str, Any],
    finetuned: dict[str, Any],
    model_path: str,
) -> dict[str, Any]:
    baseline_overall = baseline["overall"]
    finetuned_overall = finetuned["overall"]

    comparison: dict[str, Any] = {
        "model_path": model_path,
        "test_brands": finetuned["test_brands"],
        "num_test_pairs": finetuned["num_test_pairs"],
        "metrics": {},
    }

    for metric_name in finetuned_overall:
        base_val = baseline_overall.get(metric_name, 0.0)
        fine_val = finetuned_overall[metric_name]
        abs_change = fine_val - base_val
        pct_change = (
            (abs_change / base_val * 100) if base_val != 0 else None
        )
        comparison["metrics"][metric_name] = {
            "pretrained": base_val,
            "finetuned": fine_val,
            "absolute_change": abs_change,
            "percent_change": pct_change,
        }

    return comparison


def main() -> None:
    model, model_path = load_finetuned_model()
    print(f"Loaded fine-tuned model from: {model_path}")

    finetuned_results = evaluate_on_test(model, TEST_BRANDS)

    finetuned_report = {
        "model_path": model_path,
        "test_brands": TEST_BRANDS,
        "num_test_pairs": finetuned_results["num_test_pairs"],
        "overall": finetuned_results["overall"],
        "per_brand": finetuned_results["per_brand"],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    finetuned_path = REPORTS_DIR / "phase4_finetuned_test_report.json"
    with open(finetuned_path, "w", encoding="utf-8") as f:
        json.dump(finetuned_report, f, indent=2)
    print(f"Fine-tuned report saved to: {finetuned_path}")

    baseline = load_baseline_report()
    comparison = create_comparison(baseline, finetuned_results, model_path)

    comparison_path = REPORTS_DIR / "phase4_pretrained_vs_finetuned.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison report saved to: {comparison_path}")

    print("\nFine-tuned metrics on test set:")
    for k, v in finetuned_results["overall"].items():
        print(f"  {k}: {v:.4f}")

    print("\nComparison (pretrained -> finetuned):")
    for metric_name, vals in comparison["metrics"].items():
        print(
            f"  {metric_name}: {vals['pretrained']:.4f} -> {vals['finetuned']:.4f} "
            f"(change: {vals['absolute_change']:+.4f}, {vals['percent_change']:+.2f}%)"
        )


if __name__ == "__main__":
    main()
