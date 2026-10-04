import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoder as CE

try:
    from src.data_split import (
        build_training_examples,
        create_split_manifest,
        labels_for_split,
        load_labels,
        save_split_manifest,
    )
    from src.preprocessing import load_brands, load_creators
except ImportError:
    from data_split import (
        build_training_examples,
        create_split_manifest,
        labels_for_split,
        load_labels,
        save_split_manifest,
    )
    from preprocessing import load_brands, load_creators

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

DEFAULT_CONFIG: dict[str, Any] = {
    "model_name": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "seed": 42,
    "batch_size": 16,
    "epochs": 3,
    "learning_rate": 2e-5,
    "max_seq_length": 512,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_datasets(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    labels = load_labels()
    brands_list = load_brands()
    creators_list = load_creators()

    brands = {b["brand_id"]: b for b in brands_list}
    creators = {c["creator_id"]: c for c in creators_list}

    manifest = create_split_manifest(labels, seed=config["seed"])
    save_split_manifest(manifest, REPORTS_DIR / "phase4_split.json")

    train_labels = labels_for_split(labels, manifest["split"]["train"]["brand_ids"])
    val_labels = labels_for_split(labels, manifest["split"]["val"]["brand_ids"])
    test_labels = labels_for_split(labels, manifest["split"]["test"]["brand_ids"])

    train_examples = build_training_examples(train_labels, brands, creators)
    val_examples = build_training_examples(val_labels, brands, creators)
    test_examples = build_training_examples(test_labels, brands, creators)

    return train_examples, val_examples, test_examples, manifest


def train(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    set_seed(cfg["seed"])
    device = get_device()

    train_examples, val_examples, test_examples, manifest = prepare_datasets(cfg)

    model = CrossEncoder(cfg["model_name"], max_length=cfg["max_seq_length"])

    tokenizer = model.tokenizer
    train_dataset = tokenizer(
        [e["query"] for e in train_examples],
        [e["passage"] for e in train_examples],
        padding=True,
        truncation=True,
        max_length=cfg["max_seq_length"],
        return_tensors="pt",
    )

    from torch.utils.data import DataLoader, TensorDataset

    labels_tensor = torch.tensor([e["label"] for e in train_examples], dtype=torch.float)
    dataset = TensorDataset(
        train_dataset["input_ids"],
        train_dataset["attention_mask"],
        labels_tensor,
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    model.model.train()
    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    total_steps = len(loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(cfg["epochs"]):
        epoch_loss = 0.0
        for batch in loader:
            input_ids, attention_mask, batch_labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = torch.nn.functional.mse_loss(logits, batch_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1}/{cfg['epochs']} - Loss: {avg_loss:.4f}")

    timestamp = datetime.now(timezone.utc).isoformat()
    artifact_dir = ARTIFACTS_DIR / f"fine_tuned_{timestamp.replace(':', '-')}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(artifact_dir))

    metadata = {
        "model_name": cfg["model_name"],
        "seed": cfg["seed"],
        "batch_size": cfg["batch_size"],
        "epochs": cfg["epochs"],
        "learning_rate": cfg["learning_rate"],
        "max_seq_length": cfg["max_seq_length"],
        "warmup_ratio": cfg["warmup_ratio"],
        "weight_decay": cfg["weight_decay"],
        "device": device,
        "timestamp": timestamp,
        "split": manifest,
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
        "num_test_examples": len(test_examples),
        "artifact_path": str(artifact_dir),
    }

    metadata_path = REPORTS_DIR / "phase4_training_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to: {artifact_dir}")
    print(f"Metadata saved to: {metadata_path}")

    return metadata


if __name__ == "__main__":
    train()
