# Phase 2.5 Annotation Workflow

## Architecture and data flow

`annotation/annotator.py` loads and validates the selected pairs, profiles, and existing labels. It builds a compact pair context, requests one structured response from the configured OpenAI-compatible endpoint, validates it, and appends successful annotations immediately. `prompts.py`, `schema.py`, `validator.py`, and `checkpoint.py` separate prompt construction, guide terminology, strict response checks, and JSONL persistence. No work starts on import.

Inputs:

- `data/annotation_pairs_v1.json` (2,000 selected pairs)
- `data/brands.json` and `data/creators.json`
- `data/labels.json` (read-only; existing annotations are skipped)
- `reports/ANNOTATION_GUIDE.md` (terminology and criteria reflected in the prompt and enums)

Outputs:

- `artifacts/annotations_v1.jsonl`: one validated annotation per completed pair
- `artifacts/annotation_failures_v1.jsonl`: exhausted pair failures; failures are not checkpointed as completed and remain pending

The workflow uses `ANNOTATION_MODEL` (default `gpt-4o-mini`), `OPENAI_API_KEY`, and optional `OPENAI_BASE_URL` (default OpenAI API URL). It uses Python's standard library and adds no provider dependency.

## Schema and reliability

Every response must contain exactly `brand_id`, `creator_id`, `label`, `reasoning`, and `reason`. Labels are `good`, `maybe`, or `poor`. The seven dimension enums match the annotation guide: niche/audience/geography/platform/campaign use `strong`, `partial`, `weak`; budget uses `fit`, `uncertain`, `mismatch`; creator size uses `fit`, `near`, `outside`. Missing or extra fields, malformed or markdown-wrapped JSON, invalid enum values, and wrong pair IDs are rejected without repair.

The workflow allows two retries after the first request (three attempts total). Exhausted failures record IDs, error type, attempt count, and error text. Successful rows are append-flushed to JSONL immediately. On restart, existing labels and validated checkpoint pairs are skipped; failure records do not mark a pair complete, so it stays pending for retry.

## Commands

Dry run loads and validates all inputs, reports labeled and pending counts, builds prompts for a small sample, and makes no API calls or annotation writes:

```bash
python -m annotation.annotator --dry-run
```

To explicitly annotate at most three pending pairs through the production path, set `OPENAI_API_KEY` and run:

```bash
python -m annotation.annotator --limit 3
```

The full job requires an explicit invocation:

```bash
python -m annotation.annotator
```

It has not been run. `data/labels.json` is never written by this workflow, and no merge is performed.

## Verification

- Production-data dry run: 2,000 selected, 400 already labeled, 1,600 pending, 3 prompts constructed, 0 LLM calls.
- `python -m unittest discover -s tests -v`: **49 passed, 0 failed**. The workflow tests use a fake client and make no network calls.

