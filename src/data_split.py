import json
import random
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

LABEL_TO_GRADE = {"poor": 0, "maybe": 1, "good": 2}


def load_labels() -> list[dict[str, Any]]:
    with open(DATA_DIR / "labels.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["labels"]


def split_brands(
    brand_ids: list[str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, list[str]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Ratios must sum to 1.0")

    rng = random.Random(seed)
    shuffled = sorted(brand_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }


def create_split_manifest(
    labels: list[dict[str, Any]],
    seed: int = 42,
) -> dict[str, Any]:
    brand_ids = sorted(set(l["brand_id"] for l in labels))
    split = split_brands(brand_ids, seed=seed)

    manifest: dict[str, Any] = {
        "seed": seed,
        "split": {},
    }

    for split_name, brands_in_split in split.items():
        pairs = [l for l in labels if l["brand_id"] in brands_in_split]
        manifest["split"][split_name] = {
            "brand_ids": brands_in_split,
            "num_brands": len(brands_in_split),
            "num_pairs": len(pairs),
        }

    return manifest


def save_split_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def labels_for_split(
    labels: list[dict[str, Any]], brand_ids: list[str]
) -> list[dict[str, Any]]:
    brand_set = set(brand_ids)
    return [l for l in labels if l["brand_id"] in brand_set]


def build_training_examples(
    labels: list[dict[str, Any]],
    brands: dict[str, dict[str, Any]],
    creators: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        from src.preprocessing import brand_to_query, creator_to_passage
    except ImportError:
        from preprocessing import brand_to_query, creator_to_passage

    examples: list[dict[str, Any]] = []
    for item in labels:
        brand = brands.get(item["brand_id"])
        creator = creators.get(item["creator_id"])
        if brand is None or creator is None:
            continue
        examples.append(
            {
                "query": brand_to_query(brand),
                "passage": creator_to_passage(creator),
                "label": LABEL_TO_GRADE.get(item["label"], 0),
                "brand_id": item["brand_id"],
                "creator_id": item["creator_id"],
            }
        )
    return examples


def main() -> None:
    try:
        from src.preprocessing import load_brands, load_creators
    except ImportError:
        from preprocessing import load_brands, load_creators

    labels = load_labels()
    brands_list = load_brands()
    creators_list = load_creators()

    brands = {b["brand_id"]: b for b in brands_list}
    creators = {c["creator_id"]: c for c in creators_list}

    manifest = create_split_manifest(labels)
    save_split_manifest(manifest, REPORTS_DIR / "phase4_split.json")

    print("Phase 4 Data Split")
    print("=" * 40)
    for split_name, info in manifest["split"].items():
        print(
            f"{split_name:>5}: {info['num_brands']} brands, {info['num_pairs']} pairs"
        )
        print(f"       brands: {', '.join(info['brand_ids'])}")

    for split_name, info in manifest["split"].items():
        split_labels = labels_for_split(labels, info["brand_ids"])
        examples = build_training_examples(split_labels, brands, creators)
        out_path = REPORTS_DIR / f"phase4_{split_name}_examples.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2)
        print(f"Saved {len(examples)} {split_name} examples to {out_path}")


if __name__ == "__main__":
    main()
