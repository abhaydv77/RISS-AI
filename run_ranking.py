import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model import load_model
from src.preprocessing import load_brands, load_creators
from src.ranker import rank_brand


def main():
    brands = load_brands()
    creators = load_creators()

    brand = brands[0]
    print(f"Brand: {brand['brand_id']} - {brand['brand_name']}")
    print()

    model = load_model()
    results = rank_brand(model, brand, creators, k=5)

    print("Top 5 creators:")
    for rank, result in enumerate(results, start=1):
        print(f"  {rank}. {result['creator_id']} (score: {result['score']:.4f})")


if __name__ == "__main__":
    main()
