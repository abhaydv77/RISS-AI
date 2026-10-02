from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


def load_model() -> CrossEncoder:
    return CrossEncoder(MODEL_NAME)


def score_pairs(
    model: CrossEncoder, pairs: list[dict[str, str]]
) -> list[dict[str, float | str]]:
    if not pairs:
        return []

    query_passage_pairs = [(p["query"], p["passage"]) for p in pairs]
    scores = model.predict(query_passage_pairs)

    results: list[dict[str, float | str]] = []
    for pair, score in zip(pairs, scores):
        results.append(
            {
                "score": float(score),
                "brand_id": pair["brand_id"],
                "creator_id": pair["creator_id"],
            }
        )
    return results
