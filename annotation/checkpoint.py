"""Append-only JSONL checkpoints and failure history."""
import json
import os
from pathlib import Path


def pair_key(record):
    return record["brand_id"], record["creator_id"]


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path} at line {line_no}: {exc}") from exc
            if not isinstance(value, dict) or not {"brand_id", "creator_id"} <= value.keys():
                raise ValueError(f"invalid record in {path} at line {line_no}: missing pair IDs")
            records.append(value)
    return records


def completed_pair_keys(path):
    return {pair_key(row) for row in read_jsonl(path)}


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

