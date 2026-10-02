from sentence_transformers import CrossEncoder

from src.model import score_pairs
from src.preprocessing import make_pairs


def rank_brand(
    model: CrossEncoder,
    brand: dict,
    creators: list[dict],
    k: int = 5,
) -> list[dict]:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    pairs = make_pairs(brand, creators)
    scored = score_pairs(model, pairs)

    scored.sort(key=lambda x: x["score"], reverse=True)

    return scored[:k]
