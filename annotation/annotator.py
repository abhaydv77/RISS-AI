"""Explicitly invoked LLM annotation command; importing this module does no work."""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .checkpoint import append_jsonl, completed_pair_keys, pair_key, read_jsonl
from .prompts import build_pair_context, build_prompt
from .validator import AnnotationValidationError, parse_and_validate

ROOT = Path(__file__).resolve().parents[1]
MAX_RETRIES = 2  # two retries after the initial attempt; at most three calls
DEFAULT_MODEL = "gpt-4o-mini"
ANNOTATION_PATH = ROOT / "data" / "annotation_pairs_v1.json"
LABELS_PATH = ROOT / "data" / "labels.json"
BRANDS_PATH = ROOT / "data" / "brands.json"
CREATORS_PATH = ROOT / "data" / "creators.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "annotations_v1.jsonl"
FAILURES_PATH = ROOT / "artifacts" / "annotation_failures_v1.jsonl"


class AnnotationError(RuntimeError):
    pass


class LLMConfigurationError(RuntimeError):
    pass


class OpenAICompatibleClient:
    """Minimal stdlib-only client for the OpenAI Chat Completions API."""
    def __init__(self, api_key, model, base_url="https://api.openai.com/v1", timeout=90):
        if not api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is required for real annotation")
        if not model.strip():
            raise LLMConfigurationError("ANNOTATION_MODEL must not be empty")
        self.api_key, self.model = api_key, model
        self.base_url, self.timeout = base_url.rstrip("/"), timeout

    def generate(self, prompt):
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read(1500).decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM connection failed: {exc.reason}") from exc
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response missing choices[0].message.content") from exc


def load_config(require_api_key):
    model = os.environ.get("ANNOTATION_MODEL", DEFAULT_MODEL).strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not model:
        raise LLMConfigurationError("ANNOTATION_MODEL must not be empty")
    if not base_url.startswith(("https://", "http://")):
        raise LLMConfigurationError("OPENAI_BASE_URL must be an HTTP(S) URL")
    if require_api_key and not api_key:
        raise LLMConfigurationError("OPENAI_API_KEY is required for real annotation")
    return {"model": model, "base_url": base_url, "api_key": api_key}


def annotate_pair(brand, creator, client, retry_count=MAX_RETRIES):
    """Return only a fully validated annotation, or raise after bounded retries."""
    context = build_pair_context(brand, creator)
    prompt = build_prompt(context)
    errors = []
    for attempt in range(retry_count + 1):
        attempt_prompt = prompt
        if errors:
            attempt_prompt += "\n\nYour prior response failed validation: " + errors[-1] + ". Return corrected JSON only."
        try:
            raw = client.generate(attempt_prompt)
            return parse_and_validate(raw, brand["brand_id"], creator["creator_id"])
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise AnnotationError(f"all {retry_count + 1} attempts failed: " + " | ".join(errors))


def _read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {path}: {exc}") from exc


def load_inputs(annotation_path=ANNOTATION_PATH, labels_path=LABELS_PATH,
                brands_path=BRANDS_PATH, creators_path=CREATORS_PATH):
    selected_doc, labels_doc = _read_json(annotation_path), _read_json(labels_path)
    brands, creators = _read_json(brands_path), _read_json(creators_path)
    for name, doc, key in (("selected pairs", selected_doc, "pairs"), ("labels", labels_doc, "labels")):
        if not isinstance(doc, dict) or not isinstance(doc.get(key), list):
            raise ValueError(f"{name} file must contain a {key!r} array")
    if not isinstance(brands, list) or not isinstance(creators, list):
        raise ValueError("brands and creators files must be JSON arrays")
    brand_map, creator_map = {}, {}
    for rows, mapping, name in ((brands, brand_map, "brand"), (creators, creator_map, "creator")):
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get(f"{name}_id"), str):
                raise ValueError(f"each {name} record must have a string {name}_id")
            ident = row[f"{name}_id"]
            if ident in mapping:
                raise ValueError(f"duplicate {name}_id {ident}")
            required = (
                {"brand_id", "brand_name", "industry", "product", "campaign_title", "campaign_goal",
                 "campaign_description", "target_audience", "target_age_range", "target_gender", "target_locations",
                 "required_creator_niches", "preferred_creator_niches", "creator_size_preference", "minimum_followers",
                 "maximum_followers", "budget", "currency", "content_types", "platforms", "tone",
                 "mandatory_requirements", "preferred_traits", "excluded_traits"}
                if name == "brand" else
                {"creator_id", "name", "primary_niche", "secondary_niches", "bio", "location", "languages", "platforms",
                 "followers", "engagement_rate", "average_views", "audience_age_range", "audience_gender_distribution",
                 "audience_locations", "content_types", "content_style", "rate_card"}
            )
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{name} {ident} missing fields: {', '.join(sorted(missing))}")
            mapping[ident] = row
    selected, labeled = selected_doc["pairs"], labels_doc["labels"]
    selected_keys, labeled_keys = set(), set()
    for pair in selected:
        if not isinstance(pair, dict) or not isinstance(pair.get("brand_id"), str) or not isinstance(pair.get("creator_id"), str):
            raise ValueError("every selected pair must contain string brand_id and creator_id")
        key = pair_key(pair)
        if key in selected_keys:
            raise ValueError(f"duplicate selected pair {key}")
        selected_keys.add(key)
        if key[0] not in brand_map or key[1] not in creator_map:
            raise ValueError(f"selected pair references unknown profile: {key}")
    for row in labeled:
        if not isinstance(row, dict) or "brand_id" not in row or "creator_id" not in row:
            raise ValueError("every existing label must contain brand_id and creator_id")
        key = pair_key(row)
        if key in labeled_keys:
            raise ValueError(f"duplicate existing label pair {key}")
        labeled_keys.add(key)
        parse_and_validate(row, *key)
    return selected_doc, brand_map, creator_map, labeled_keys


def _validate_checkpoint(path):
    rows = read_jsonl(path)
    keys = set()
    for row in rows:
        key = pair_key(row)
        if key in keys:
            raise ValueError(f"duplicate completed checkpoint pair {key}")
        parse_and_validate(row, *key)
        keys.add(key)
    return keys


def run_job(client, *, limit=None, dry_run=False, sample_size=3,
            annotation_path=ANNOTATION_PATH, labels_path=LABELS_PATH,
            brands_path=BRANDS_PATH, creators_path=CREATORS_PATH,
            checkpoint_path=CHECKPOINT_PATH, failures_path=FAILURES_PATH):
    selected_doc, brands, creators, labeled = load_inputs(annotation_path, labels_path, brands_path, creators_path)
    selected = selected_doc["pairs"]
    checkpointed = _validate_checkpoint(checkpoint_path)
    selected_keys = {pair_key(p) for p in selected}
    labeled_selected = labeled & selected_keys
    pending = [p for p in selected if pair_key(p) not in labeled_selected and pair_key(p) not in checkpointed]
    if dry_run:
        if limit is not None:
            sample_size = min(sample_size, limit)
        sample = pending[:max(0, sample_size)]
        for pair in sample:
            prompt = build_prompt(build_pair_context(brands[pair["brand_id"]], creators[pair["creator_id"]]))
            if not prompt.strip():
                raise ValueError(f"empty prompt for {pair_key(pair)}")
        print(f"Total selected pairs: {len(selected)}")
        print(f"Already labeled: {len(labeled_selected)}")
        print(f"Pending: {len(pending)}")
        print(f"Configuration valid: model={os.environ.get('ANNOTATION_MODEL', DEFAULT_MODEL).strip()}")
        print(f"Dry-run sample: {len(sample)} pairs")
        print("LLM calls: 0")
        return {"selected":len(selected),"labeled":len(labeled_selected),"pending":len(pending),"sample":len(sample),"calls":0}

    work = pending if limit is None else pending[:limit]
    successes = failures = calls = 0
    total = len(work)
    for index, pair in enumerate(work, 1):
        bid, cid = pair_key(pair)
        print(f"Annotating {index}/{total}\nPair: {bid}/{cid}")
        try:
            # Calls count is tracked even if a request itself raises.
            pair_calls = 0
            class CountingClient:
                def generate(self, prompt):
                    nonlocal calls, pair_calls
                    calls += 1
                    pair_calls += 1
                    return client.generate(prompt)
            result = annotate_pair(brands[bid], creators[cid], CountingClient())
            append_jsonl(checkpoint_path, result)
            successes += 1
            print(f"Success: {result['label']}")
        except Exception as exc:
            failures += 1
            err_type = "validation_error" if isinstance(exc, AnnotationValidationError) or "AnnotationValidationError" in str(exc) else type(exc).__name__
            failure = {"brand_id":bid,"creator_id":cid,"error_type":err_type,"attempts":pair_calls,"error":str(exc)}
            append_jsonl(failures_path, failure)
            print(f"Failed after {failure['attempts']} attempts: {failure['error']}", file=sys.stderr)
        if index % 50 == 0 or index == total:
            print(f"Progress: {index}/{total}\nSuccess: {successes}\nFailed: {failures}")
    return {"processed":total,"success":successes,"failed":failures,"calls":calls}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resumable LLM-assisted pair annotation")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and preview prompts without API calls or writes")
    parser.add_argument("--limit", type=int, help="process at most this many pending pairs")
    parser.add_argument("--sample-size", type=int, default=3, help="number of pending pairs to inspect in dry-run")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.sample_size < 0:
        parser.error("--sample-size cannot be negative")
    try:
        config = load_config(require_api_key=not args.dry_run)
        if args.dry_run:
            client = None
        else:
            client = OpenAICompatibleClient(config["api_key"], config["model"], config["base_url"])
        run_job(
            client, limit=args.limit, dry_run=args.dry_run, sample_size=args.sample_size,
            annotation_path=ANNOTATION_PATH, labels_path=LABELS_PATH,
            brands_path=BRANDS_PATH, creators_path=CREATORS_PATH,
            checkpoint_path=CHECKPOINT_PATH, failures_path=FAILURES_PATH,
        )
    except (ValueError, LLMConfigurationError, AnnotationError) as exc:
        print(f"Annotation stopped: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
