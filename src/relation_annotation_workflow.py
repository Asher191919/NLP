"""Prepare and validate a blind, text-aligned relation annotation packet.

The packet is selected from unused LLMs4OL 2026 training samples using text-only
relation cues. Official gold and model predictions are excluded from annotator
files. Two independent annotations are compared before a separate adjudication
file can be frozen for calibration.

Two provenance modes share the same packet geometry:

- `human` (default, protocol `relation_text_annotation_v1`): three distinct people,
  `annotator_type="human"`. Frozen output is `confirmatory_eligible=True`.
- `model` (protocol `relation_text_model_annotation_v1`): dual-model weak labelling
  with model adjudication. Frozen output is `annotation_source=
  "dual_model_adjudicated"`, `human_annotated=False`,
  `confirmatory_eligible=False`, and `calibration_ready=False` so it can only feed
  exploratory calibration, never the confirmatory gate.

Provenance is validated against the requested mode. A model file may not declare
`annotator_type="human"`, and a human file may not declare `annotator_type="model"`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import run_official_blind_eval as blind
from ontobridg.relations.canonicalization import (
    build_entity_inventory,
    canonicalize_predicate,
    canonicalize_triple,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "relation_text_annotation_v1"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "annotations" / PROTOCOL

# Model provenance mode. Same packet geometry, different declared provenance.
# Model labels are weak labels: they may support exploratory calibration but can
# never open the confirmatory gate, and must never be written as human labels.
MODEL_PROTOCOL = "relation_text_model_annotation_v1"
MODEL_DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "annotations" / MODEL_PROTOCOL
ANNOTATION_SOURCES = ("human", "model")
MODEL_ANNOTATION_SOURCE_ID = "dual_model_adjudicated"
MODEL_CLAIM_LABEL = "exploratory dual-LLM weak-label calibration"
TEMPLATE_ANNOTATOR_IDS = {
    "annotator_a",
    "annotator_b",
    "model_annotator_a",
    "model_annotator_b",
}
MODEL_ANNOTATOR_FILE_STEMS = {
    "annotator_a": "model_annotator_a",
    "annotator_b": "model_annotator_b",
}
DEFAULT_EXCLUSION_RESULTS = (
    ROOT
    / "experiments"
    / "results"
    / "official_stratified_open_openai_ab_s5_seed20260922.json",
    ROOT
    / "experiments"
    / "results"
    / "relation_calibration_a_n27_seed20260923.json",
    ROOT
    / "experiments"
    / "results"
    / "relation_calibration_ab_n12_seed20260923.json",
    ROOT
    / "experiments"
    / "results"
    / "relation_calibration_open_vocab_guarded_v3_relations_a_indices938-2077-2424_seed20260923.json",
)
SOURCE_PATHS = {
    "a": blind.TASK_CONFIG["a"]["path"],
    "b": blind.TASK_CONFIG["b"]["path"],
}
SYMMETRIC_PREDICATES = {"disjoint with", "equivalent class"}
CORE_PREDICATES = {*blind.INSTANCE_RELATIONS, *blind.TAXONOMY_RELATIONS}
PREFERRED_PREDICATES = (
    "part_of",
    "has part",
    "develops_from",
    "derives from",
    "located in",
    "positively regulates",
    "regulates",
)
TRIGGER_PATTERNS = {
    "part_whole": re.compile(
        r"\b(?:part of|consists? of|composed of|comprises?|contains?)\b",
        re.IGNORECASE,
    ),
    "origin": re.compile(
        r"\b(?:deriv(?:e|ed|es) from|develops? from|arises? from)\b",
        re.IGNORECASE,
    ),
    "location": re.compile(r"\b(?:located in|found in|occurs? in)\b", re.IGNORECASE),
    "regulation": re.compile(
        r"\b(?:regulat(?:e|es|ed|ing)|inhibit(?:s|ed|ing)?|"
        r"activat(?:e|es|ed|ing)|controls?)\b",
        re.IGNORECASE,
    ),
    "interaction": re.compile(r"\binteracts? with\b", re.IGNORECASE),
    "association": re.compile(r"\bassociated with\b", re.IGNORECASE),
    "causal": re.compile(r"\bcauses?\b", re.IGNORECASE),
    "division": re.compile(r"\bdivided into\b", re.IGNORECASE),
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _atomic_write_text(path, text)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: each row must be an object")
        rows.append(row)
    return rows


def relation_trigger_features(context: str) -> dict[str, Any]:
    family_counts = {
        family: len(pattern.findall(context))
        for family, pattern in TRIGGER_PATTERNS.items()
    }
    active = sorted(family for family, count in family_counts.items() if count)
    return {
        "trigger_families": active,
        "trigger_family_count": len(active),
        "trigger_occurrences": sum(family_counts.values()),
        "trigger_counts": {family: family_counts[family] for family in active},
    }


def _result_selected_indices(path: Path) -> dict[str, set[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = {"a": set(), "b": set()}
    for task in ("a", "b"):
        task_payload = payload.get("tasks", {}).get(f"task_{task}", {})
        indices = task_payload.get("meta", {}).get("selected_indices")
        if indices is None:
            systems = task_payload.get("systems", {})
            first_system = next(iter(systems.values()), {})
            indices = [
                row.get("sample_index")
                for row in first_system.get("samples", [])
                if row.get("sample_index") is not None
            ]
        selected[task].update(int(index) for index in indices or [])
    return selected


def build_exclusions(
    *, prefix: int = 30, result_paths: Iterable[Path] = DEFAULT_EXCLUSION_RESULTS
) -> tuple[dict[str, set[int]], list[dict[str, Any]]]:
    excluded = {"a": set(range(max(0, prefix))), "b": set(range(max(0, prefix)))}
    sources: list[dict[str, Any]] = [
        {"kind": "blind_prefix", "exclusive_stop": max(0, prefix)}
    ]
    for path in result_paths:
        if not path.exists():
            raise FileNotFoundError(f"required exclusion result does not exist: {path}")
        selected = _result_selected_indices(path)
        for task in ("a", "b"):
            excluded[task].update(selected[task])
        sources.append(
            {
                "kind": "result_artifact",
                "path": str(path),
                "sha256": _file_sha256(path),
                "task_a_indices": sorted(selected["a"]),
                "task_b_indices": sorted(selected["b"]),
            }
        )
    return excluded, sources


def select_annotation_samples(
    indexed_samples: list[tuple[int, dict[str, Any]]],
    *,
    task: str,
    count: int,
    seed: int,
    excluded_indices: set[int],
    max_context_chars: int = 6000,
) -> tuple[list[tuple[int, dict[str, Any]]], dict[int, dict[str, Any]], dict[str, Any]]:
    candidates: list[tuple[Any, ...]] = []
    family_pool = Counter()
    for index, sample in indexed_samples:
        if index in excluded_indices:
            continue
        context = str(sample.get("context") or "")[:max_context_chars]
        features = relation_trigger_features(context)
        if not features["trigger_families"]:
            continue
        family_pool.update(features["trigger_families"])
        sample_id = str(sample.get("id") or index)
        rank = _sha256_text(f"{seed}:{task}:{sample_id}")
        candidates.append(
            (
                -int(features["trigger_family_count"]),
                -int(features["trigger_occurrences"]),
                abs(len(context) - 1800),
                rank,
                index,
                sample,
                features,
            )
        )
    candidates.sort(key=lambda row: row[:5])
    if len(candidates) < count:
        raise ValueError(
            f"task {task} has {len(candidates)} unused text-triggered samples; need {count}"
        )
    chosen = candidates[:count]
    selected = sorted(((row[4], row[5]) for row in chosen), key=lambda row: row[0])
    metadata = {
        row[4]: {
            **row[6],
            "selection_rank_sha256": row[3],
            "context_chars": min(
                len(str(row[5].get("context") or "")), max_context_chars
            ),
            "context_truncated": len(str(row[5].get("context") or ""))
            > max_context_chars,
        }
        for row in chosen
    }
    summary = {
        "eligible_text_triggered_samples": len(candidates),
        "excluded_indices_count": len(excluded_indices),
        "selected_samples": len(selected),
        "eligible_trigger_family_counts": dict(sorted(family_pool.items())),
        "selected_trigger_family_counts": dict(
            sorted(
                Counter(
                    family
                    for index, _ in selected
                    for family in metadata[index]["trigger_families"]
                ).items()
            )
        ),
    }
    return selected, metadata, summary


def _sample_key(seed: int, task: str, index: int, sample_id: str) -> str:
    return _sha256_text(f"{PROTOCOL}:{seed}:{task}:{index}:{sample_id}")[:20]


def _legal_inputs(sample: dict[str, Any], *, task: str, max_context_chars: int) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "context": str(sample.get("context") or "")[:max_context_chars]
    }
    if task == "b":
        inputs.update(
            {
                "terms": sample.get("terms") or [],
                "types": sample.get("types") or [],
                "initial-primitive-ontology-triples": sample.get(
                    "initial-primitive-ontology-triples"
                )
                or [],
            }
        )
    return inputs


def build_packet(
    *,
    count_per_task: int = 15,
    seed: int = 20260923,
    max_context_chars: int = 6000,
    prefix: int = 30,
    result_paths: Iterable[Path] = DEFAULT_EXCLUSION_RESULTS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result_paths = tuple(result_paths)
    exclusions, exclusion_sources = build_exclusions(
        prefix=prefix, result_paths=result_paths
    )
    selected_by_task: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    metadata_by_task: dict[str, dict[int, dict[str, Any]]] = {}
    pools: dict[str, Any] = {}
    for task in ("a", "b"):
        selected, metadata, pool = select_annotation_samples(
            blind.load_samples(task, 0, 0),
            task=task,
            count=count_per_task,
            seed=seed,
            excluded_indices=exclusions[task],
            max_context_chars=max_context_chars,
        )
        selected_by_task[task] = selected
        metadata_by_task[task] = metadata
        pool["selected_overlap_with_exclusions"] = sorted(
            {index for index, _ in selected} & exclusions[task]
        )
        pools[f"task_{task}"] = pool

    packet_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for task in ("a", "b"):
        for index, sample in selected_by_task[task]:
            sample_id = str(sample.get("id") or index)
            key = _sample_key(seed, task, index, sample_id)
            inputs = _legal_inputs(
                sample, task=task, max_context_chars=max_context_chars
            )
            manifest_rows.append(
                {
                    "sample_key": key,
                    "task": task,
                    "source_index": index,
                    "source_sample_id_sha256": _sha256_text(sample_id),
                    "context_sha256": _sha256_text(inputs["context"]),
                    "legal_input_fields": list(inputs),
                    **metadata_by_task[task][index],
                }
            )
            packet_rows.append(
                {
                    "schema_version": PROTOCOL,
                    "sample_key": key,
                    "task": task,
                    "inputs": inputs,
                    "annotation": {
                        "annotator_id": "",
                        "annotator_type": "",
                        "completed_at": "",
                        "status": "pending",
                        "relations": [],
                        "sample_notes": "",
                    },
                }
            )
    manifest_basis = {
        "protocol": PROTOCOL,
        "seed": seed,
        "count_per_task": count_per_task,
        "max_context_chars": max_context_chars,
        "samples": manifest_rows,
    }
    packet_id = _sha256_text(
        json.dumps(manifest_basis, ensure_ascii=False, sort_keys=True)
    )[:20]
    manifest = {
        "meta": {
            "protocol": PROTOCOL,
            "packet_id": packet_id,
            "seed": seed,
            "tasks": ["a", "b"],
            "count_per_task": count_per_task,
            "n_samples": len(packet_rows),
            "max_context_chars": max_context_chars,
            "selection_uses_official_gold": False,
            "selection_uses_model_output": False,
            "annotator_gold_visible": False,
            "annotator_model_output_visible": False,
            "requires_two_independent_annotators": True,
            "requires_adjudication": True,
            "calibration_ready": False,
        },
        "source_files": [
            {"task": task, "path": str(path), "sha256": _file_sha256(path)}
            for task, path in SOURCE_PATHS.items()
        ],
        "exclusion_sources": exclusion_sources,
        "pools": pools,
        "preferred_predicates": list(PREFERRED_PREDICATES),
        "samples": manifest_rows,
    }
    return manifest, packet_rows


GUIDELINES = """# Relation annotation guidelines

This packet is for exhaustive, text-grounded non-taxonomic relation annotation.
Annotators must work independently and must not inspect official gold, model output,
confidence scores, or the other annotator's file.

For every relation entailed by the visible inputs:

1. Record `subject`, canonical `predicate`, and `object`.
2. Prefer: `part_of`, `has part`, `develops_from`, `derives from`, `located in`,
   `positively regulates`, or `regulates`. Other specific non-taxonomic predicates
   are allowed when none of these is faithful.
3. Copy one exact evidence quote for an explicit relation. Use two exact quotes and
   `cross_sentence` only when the relation requires two nearby spans.
4. Both endpoint strings must occur in the combined evidence after case and
   whitespace normalization. Use the surface forms actually stated in the text.
5. Do not annotate `is-a`, subclass, instance/type, or ordinary co-occurrence.
6. Do not infer disjointness from ordinary sibling classes or equivalence from an
   ordinary subtype statement.
7. Set sample status to `complete` only after checking the full visible input. Use
   `abstain` when the text cannot be judged reliably; never treat abstention as a
   negative label.
8. For a completed row, set a stable pseudonymous `annotator_id`, set
   `annotator_type` to `human`, and record an ISO-8601 `completed_at` timestamp.
   Replace the template placeholder ID. The two annotators and the adjudicator
   must use three different IDs.

Relation object shape:

```json
{"subject":"...","predicate":"part_of","object":"...","evidence_mode":"explicit","evidence":["exact quote"],"notes":""}
```

After both files are complete, `compare` creates a separate adjudication file.
Only a human adjudicator may resolve disagreements and mark each sample `complete`.
`freeze` rejects pending, abstained, malformed, or unresolved samples.
"""


def prepare_packet(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    manifest, packet_rows = build_packet(**kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("meta", {}).get("packet_id") != manifest["meta"]["packet_id"]:
            raise ValueError(
                f"{manifest_path} contains a different packet; choose a new output directory"
            )
    _atomic_write_json(manifest_path, manifest)
    _atomic_write_text(output_dir / "annotation_guidelines.md", GUIDELINES)
    for annotator in ("annotator_a", "annotator_b"):
        path = output_dir / f"{annotator}.jsonl"
        if path.exists():
            continue
        rows = json.loads(json.dumps(packet_rows, ensure_ascii=False))
        for row in rows:
            row["annotation"]["annotator_id"] = annotator
        _write_jsonl(path, rows)
    return manifest


MODEL_PROMPT_VERSION = "relation_model_annotation_v1"

MODEL_ANNOTATOR_PROMPT = """You are an independent text relation annotator. You may use only the
`inputs` field of the current JSON line. You must not look at official gold, model candidates,
confidence scores, caches, or the other annotator's output.

Enumerate every non-taxonomic relation that `inputs.context` states explicitly or entails
reliably. Do not annotate is-a, subclass, instance/type, or ordinary co-occurrence. Evidence
must be copied verbatim; use one evidence span for explicit and two spans for cross_sentence.
Both relation endpoints must appear in the evidence. Do not guess unsupported relations.

Keep `inputs` and `sample_key` exactly unchanged and update only `annotation`.
`annotator_type` must be `model`. Output exactly one valid JSON line per sample, in input
order, with no Markdown and no explanation."""

MODEL_ADJUDICATOR_PROMPT = """You are the adjudicator for two independent model annotations.
Normalize both relation sets to (subject, predicate, object) before comparing. For relations
the two annotators agree on, still check evidence and direction. Rule every disagreement as
`entailed` or `not_entailed`. Read each text yourself to check for relations both annotators
missed. Record your model, prompt version, timestamp, and the reason for every decision."""

MODEL_GUIDELINES = """# Relation annotation guidelines (dual-model weak labelling)

Protocol: `relation_text_model_annotation_v1`
Prompt version: `relation_model_annotation_v1`

This packet is for exhaustive, text-grounded non-taxonomic relation annotation.
Model annotators must work in isolated contexts and must not inspect official gold,
model output, confidence scores, caches, or the other annotator's file.

## Per-sample rules

1. Read the whole `inputs.context`.
2. Enumerate every non-taxonomic relation the text states explicitly or entails
   reliably.
3. Do not annotate `is-a`, subclass, instance/type, or ordinary co-occurrence.
4. Prefer these canonical predicates: `part_of`, `has part`, `develops_from`,
   `derives from`, `located in`, `positively regulates`, `regulates`.
5. Use a different predicate only when none of the above is accurate.
6. Check direction strictly: `X part_of Y` must not be written `Y part_of X`.
7. `explicit` must carry one verbatim evidence quote.
8. `cross_sentence` must carry two verbatim evidence quotes.
9. `subject` and `object` must both occur in the evidence after ignoring case and
   collapsing repeated whitespace.
10. When no relation holds, emit an empty array. Never guess.
11. For Task B, `terms`, `types`, and `initial-primitive-ontology-triples` may only
    help locate entities. They are never evidence.

## Model annotation shape

Use `annotator_type: "model"`. Never write `hostname: "human"`.

```json
{
  "annotator_id": "model_a_<model-id>",
  "annotator_type": "model",
  "model_id": "<exact-model-name>",
  "prompt_version": "relation_model_annotation_v1",
  "completed_at": "2026-09-24T12:00:00+08:00",
  "status": "complete",
  "relations": [
    {
      "subject": "...",
      "predicate": "part_of",
      "object": "...",
      "evidence_mode": "explicit",
      "evidence": ["verbatim quote"],
      "notes": ""
    }
  ],
  "sample_notes": ""
}
```

## Independence and claim limits

- Give `model_annotator_a.jsonl` to annotator A and `model_annotator_b.jsonl` to
  annotator B. Neither may read the other file.
- Annotator C may read both files only after A and B are finished.
- Prefer different model families. Three same-origin agents are repeated sampling,
  not strong independence; record the real model IDs and say so.
- Use temperature 0 and record model name, version, date, and prompt hash.
- These are **weak labels**. They support exploratory engineering calibration. They
  do **not** support an independently human-annotated claim, and the confirmatory
  gate stays closed until a real human audit exists.

## Adjudication

1. Normalize both relation sets; match on the full `(subject, predicate, object)`
   triple.
2. Agreement still needs evidence and direction checked by C.
3. Judge every disagreement `entailed` or `not_entailed`.
4. C must also read each text to catch relations both annotators missed.
5. C records its own model, prompt, timestamp, and decision reasons.
6. After freezing, report sample-level exact agreement, relation-set Jaccard,
   positive agreement, and predicate-level disagreement.

## Lowest-cost credible compromise

Have A and B label all samples, have C adjudicate, then have **one human** review all
disagreements, all positives, and a random 20% of agreed negatives. Report that as
`dual-LLM annotation with targeted human audit`. That is far cheaper than full
three-person annotation and far more defensible than an unaudited model self-label.
"""


def prepare_model_packet(
    output_dir: Path,
    *,
    source_packet_dir: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Prepare a dual-model weak-labelling packet.

    Reuses the same deterministic sample selection as the human packet so that the
    two label sets are directly comparable and the guarded-v2 generation cache can
    be replayed for zero further API calls.
    """
    manifest, packet_rows = build_packet(**kwargs)
    manifest["meta"].update(
        {
            "protocol": MODEL_PROTOCOL,
            "sample_source_protocol": PROTOCOL,
            "annotation_source": "model",
            "annotator_type": "model",
            "prompt_version": MODEL_PROMPT_VERSION,
            "requires_two_independent_model_annotators": True,
            "requires_model_adjudication": True,
            "human_annotated": False,
            "confirmatory_eligible": False,
            "calibration_ready": False,
            "exploratory_calibration_ready": True,
            "calibration_scope": "exploratory_dual_llm_weak_label",
            "claim_label": MODEL_CLAIM_LABEL,
            "independence_caveat": (
                "declared_model_ids_are_self_reported; same-origin agents give "
                "repeated sampling rather than strong independence"
            ),
            "human_audit_scope": (
                "all_disagreements, all_positive_relations, "
                "random_20pct_agreed_negatives"
            ),
        }
    )
    if source_packet_dir is not None:
        source_manifest = Path(source_packet_dir) / "manifest.json"
        if source_manifest.exists():
            manifest["meta"]["sample_source_manifest_sha256"] = _file_sha256(
                source_manifest
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / "manifest.json", manifest)
    _atomic_write_text(output_dir / "annotation_guidelines.md", MODEL_GUIDELINES)
    for annotator, stem in MODEL_ANNOTATOR_FILE_STEMS.items():
        path = output_dir / f"{stem}.jsonl"
        if path.exists():
            continue
        rows = json.loads(json.dumps(packet_rows, ensure_ascii=False))
        for row in rows:
            row["schema_version"] = MODEL_PROTOCOL
            row["annotation"] = {
                "annotator_id": stem,
                "annotator_type": "",
                "model_id": "",
                "prompt_version": "",
                "completed_at": "",
                "status": "pending",
                "relations": [],
                "sample_notes": "",
            }
        _write_jsonl(path, rows)
    return manifest


def _compact(value: Any) -> str:
    return " ".join(str(value).lower().strip().split())


def _relation_key(
    relation: dict[str, Any], *, context: str, inventory: list[str]
) -> tuple[str, str, str]:
    canonical, _ = canonicalize_triple(
        [relation.get("subject"), relation.get("predicate"), relation.get("object")],
        context=context,
        entity_inventory=inventory,
    )
    subject, predicate, obj = canonical
    if predicate in SYMMETRIC_PREDICATES and obj < subject:
        subject, obj = obj, subject
    return subject, predicate, obj


def annotation_type_for_source(annotation_source: str) -> str:
    if annotation_source not in ANNOTATION_SOURCES:
        raise ValueError(
            f"annotation_source must be one of {ANNOTATION_SOURCES}, got {annotation_source!r}"
        )
    return annotation_source


def validate_annotation_rows(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    annotation_source: str = "human",
) -> list[str]:
    expected_type = annotation_type_for_source(annotation_source)
    errors: list[str] = []
    manifest_by_key = {row["sample_key"]: row for row in manifest["samples"]}
    expected = set(manifest_by_key)
    seen: set[str] = set()
    for row_number, row in enumerate(rows, 1):
        key = str(row.get("sample_key") or "")
        label = f"row {row_number} ({key or 'missing-key'})"
        if key not in expected:
            errors.append(f"{label}: unknown sample_key")
            continue
        if key in seen:
            errors.append(f"{label}: duplicate sample_key")
        seen.add(key)
        manifest_row = manifest_by_key[key]
        if manifest_row.get("task") is not None and row.get("task") != manifest_row.get("task"):
            errors.append(f"{label}: task does not match manifest")
        annotation = row.get("annotation")
        if not isinstance(annotation, dict):
            errors.append(f"{label}: annotation must be an object")
            continue
        status = annotation.get("status")
        if status not in {"pending", "complete", "abstain"}:
            errors.append(f"{label}: invalid status {status!r}")
        if status == "complete":
            annotator_id = str(annotation.get("annotator_id") or "").strip()
            if not annotator_id:
                errors.append(f"{label}: complete annotation requires annotator_id")
            elif annotator_id in TEMPLATE_ANNOTATOR_IDS:
                errors.append(f"{label}: replace the template annotator_id")
            declared_type = annotation.get("annotator_type")
            # The declared type must match the requested source exactly. A model
            # mode file may not claim "human", and a human mode file may not claim
            # "model" - provenance is validated, never inferred or upgraded.
            if declared_type != expected_type:
                errors.append(
                    f"{label}: complete annotation requires "
                    f"annotator_type={expected_type!r} (found {declared_type!r})"
                )
            if not str(annotation.get("completed_at") or "").strip():
                errors.append(f"{label}: complete annotation requires completed_at")
            if expected_type == "model":
                for field in ("model_id", "prompt_version"):
                    if not str(annotation.get(field) or "").strip():
                        errors.append(
                            f"{label}: model annotation requires {field}"
                        )
        inputs = row.get("inputs") or {}
        if not isinstance(inputs, dict):
            errors.append(f"{label}: inputs must be an object")
            inputs = {}
        legal_fields = manifest_row.get("legal_input_fields")
        if legal_fields is not None and set(inputs) != set(legal_fields):
            errors.append(f"{label}: input fields do not match manifest")
        context = str(inputs.get("context") or "")
        expected_context_hash = manifest_row.get("context_sha256")
        if expected_context_hash and _sha256_text(context) != expected_context_hash:
            errors.append(f"{label}: context hash does not match manifest")
        evidence_haystack = context
        inventory = build_entity_inventory(inputs, task=str(row.get("task") or "a"))
        relation_keys: set[tuple[str, str, str]] = set()
        relations = annotation.get("relations")
        if not isinstance(relations, list):
            errors.append(f"{label}: relations must be a list")
            continue
        if status != "complete" and relations:
            errors.append(f"{label}: only complete samples may contain relations")
        for relation_number, relation in enumerate(relations, 1):
            relation_label = f"{label} relation {relation_number}"
            if not isinstance(relation, dict):
                errors.append(f"{relation_label}: relation must be an object")
                continue
            values = [
                relation.get("subject"),
                relation.get("predicate"),
                relation.get("object"),
            ]
            if any(not str(value or "").strip() for value in values):
                errors.append(f"{relation_label}: subject, predicate, and object are required")
                continue
            predicate, _, _ = canonicalize_predicate(relation["predicate"])
            if predicate in CORE_PREDICATES:
                errors.append(f"{relation_label}: core typing/taxonomy predicates are forbidden")
            evidence = relation.get("evidence")
            if not isinstance(evidence, list) or len(evidence) not in {1, 2}:
                errors.append(f"{relation_label}: evidence must contain one or two quotes")
                continue
            mode = relation.get("evidence_mode")
            expected_mode = "explicit" if len(evidence) == 1 else "cross_sentence"
            if mode != expected_mode:
                errors.append(
                    f"{relation_label}: evidence_mode must be {expected_mode!r}"
                )
            missing_quotes = [quote for quote in evidence if str(quote) not in evidence_haystack]
            if missing_quotes:
                errors.append(f"{relation_label}: evidence quote is not verbatim")
            joined = _compact(" ".join(str(quote) for quote in evidence))
            for endpoint_name in ("subject", "object"):
                endpoint = _compact(relation[endpoint_name])
                if endpoint and endpoint not in joined:
                    errors.append(
                        f"{relation_label}: {endpoint_name} is absent from combined evidence"
                    )
            relation_key = _relation_key(
                relation, context=context, inventory=inventory
            )
            if relation_key in relation_keys:
                errors.append(f"{relation_label}: duplicate canonical relation")
            relation_keys.add(relation_key)
    missing = sorted(expected - seen)
    if missing:
        errors.append(f"missing {len(missing)} manifest samples")
    return errors


def validate_annotation_file(
    path: Path, manifest: dict[str, Any], *, annotation_source: str = "human"
) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    errors = validate_annotation_rows(
        rows, manifest, annotation_source=annotation_source
    )
    if errors:
        raise ValueError("\n".join(errors))
    return rows


def _copy_inputs(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(row["inputs"], ensure_ascii=False))


def _declared_provenance(annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": annotation.get("annotator_id"),
        "type": annotation.get("annotator_type"),
        "model_id": annotation.get("model_id"),
        "prompt_version": annotation.get("prompt_version"),
        "completed_at": annotation.get("completed_at"),
    }


def compare_annotations(
    *,
    manifest_path: Path,
    annotation_a_path: Path,
    annotation_b_path: Path,
    output_path: Path,
    annotation_source: str = "human",
    agreement_path: Path | None = None,
) -> dict[str, Any]:
    annotation_type_for_source(annotation_source)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_a = validate_annotation_file(
        annotation_a_path, manifest, annotation_source=annotation_source
    )
    rows_b = validate_annotation_file(
        annotation_b_path, manifest, annotation_source=annotation_source
    )
    by_a = {row["sample_key"]: row for row in rows_a}
    by_b = {row["sample_key"]: row for row in rows_b}
    output_rows: list[dict[str, Any]] = []
    counts = Counter()
    for manifest_row in manifest["samples"]:
        key = manifest_row["sample_key"]
        row_a = by_a[key]
        row_b = by_b[key]
        if row_a.get("inputs") != row_b.get("inputs"):
            raise ValueError(f"sample {key}: annotator inputs differ")
        status_a = row_a["annotation"]["status"]
        status_b = row_b["annotation"]["status"]
        annotator_a = str(row_a["annotation"].get("annotator_id") or "").strip()
        annotator_b = str(row_b["annotation"].get("annotator_id") or "").strip()
        if status_a == status_b == "complete" and annotator_a == annotator_b:
            raise ValueError(
                f"sample {key}: independent annotations require different annotator IDs"
            )
        context = str(row_a["inputs"].get("context") or "")
        inventory = build_entity_inventory(row_a["inputs"], task=row_a["task"])

        def relation_map(row: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
            return {
                _relation_key(relation, context=context, inventory=inventory): relation
                for relation in row["annotation"]["relations"]
            }

        relations_a = relation_map(row_a) if status_a == "complete" else {}
        relations_b = relation_map(row_b) if status_b == "complete" else {}
        agreed_keys = sorted(set(relations_a) & set(relations_b))
        disagreement_keys = sorted(set(relations_a) ^ set(relations_b))
        ready = status_a == status_b == "complete"
        counts["samples"] += 1
        counts["both_complete"] += int(ready)
        counts["agreed_relations"] += len(agreed_keys)
        counts["disagreements"] += len(disagreement_keys)
        output_rows.append(
            {
                "schema_version": PROTOCOL,
                "packet_id": manifest["meta"]["packet_id"],
                "sample_key": key,
                "task": row_a["task"],
                "inputs": _copy_inputs(row_a),
                "source_annotation_status": {
                    "annotator_a": status_a,
                    "annotator_b": status_b,
                },
                "source_annotation_provenance": {
                    "annotator_a": _declared_provenance(row_a["annotation"]),
                    "annotator_b": _declared_provenance(row_b["annotation"]),
                },
                "agreed_relations": [
                    {
                        "triple": list(relation_key),
                        "annotator_a": relations_a[relation_key],
                        "annotator_b": relations_b[relation_key],
                    }
                    for relation_key in agreed_keys
                ],
                "disagreements": [
                    {
                        "triple": list(relation_key),
                        "annotator_a": relations_a.get(relation_key),
                        "annotator_b": relations_b.get(relation_key),
                        "decision": "pending",
                        "adjudicated_evidence_mode": "",
                        "adjudicated_evidence": [],
                        "notes": "",
                    }
                    for relation_key in disagreement_keys
                ],
                "adjudication": {
                    "adjudicator_id": "",
                    "adjudicator_type": "",
                    "adjudicator_model_id": "",
                    "adjudicator_prompt_version": "",
                    "completed_at": "",
                    "status": "pending",
                    "accept_agreed_relations": ready,
                    "sample_notes": "",
                },
            }
        )
    _write_jsonl(output_path, output_rows)
    agreement = summarize_agreement(output_rows)
    agreement["protocol"] = manifest["meta"].get("protocol")
    agreement["packet_id"] = manifest["meta"]["packet_id"]
    agreement["annotation_source"] = annotation_source
    if agreement_path is not None:
        _atomic_write_json(agreement_path, agreement)
    counts.update(
        {
            "sample_exact_agreement_rate": agreement["sample_exact_agreement_rate"],
            "relation_set_jaccard_mean": agreement["relation_set_jaccard_mean"],
            "positive_agreement_rate": agreement["positive_agreement_rate"],
        }
    )
    counts["agreement"] = agreement
    return dict(counts)


def summarize_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Inter-annotator agreement over adjudication rows.

    Metrics: sample-level exact agreement, macro/micro relation-set Jaccard,
    standard positive agreement (Dice), and a predicate-level disagreement
    histogram.
    """
    samples = len(rows)
    exact = 0
    jaccard_total = 0.0
    per_task: dict[str, dict[str, Any]] = {}
    asserted = 0
    asserted_a_total = 0
    asserted_b_total = 0
    only_a_total = 0
    only_b_total = 0
    agreed_total = 0
    predicate_disagreements: Counter = Counter()
    for row in rows:
        task = str(row.get("task") or "?")
        bucket = per_task.setdefault(
            task,
            {
                "samples": 0,
                "exact": 0,
                "jaccard_sum": 0.0,
                "asserted": 0,
                "asserted_a": 0,
                "asserted_b": 0,
                "only_a": 0,
                "only_b": 0,
                "agreed": 0,
            },
        )
        agreed = {tuple(item["triple"]) for item in row.get("agreed_relations") or []}
        only_a: set[tuple[str, str, str]] = set()
        only_b: set[tuple[str, str, str]] = set()
        for item in row.get("disagreements") or []:
            triple = tuple(item.get("triple") or [])
            if len(triple) != 3:
                continue
            asserted_a = bool(item.get("annotator_a"))
            asserted_b = bool(item.get("annotator_b"))
            if asserted_a:
                only_a.add(triple)
            if asserted_b:
                only_b.add(triple)
            # Count a predicate disagreement only when at least one annotator
            # actually asserted it. Rows added solely by the adjudicator (joint
            # misses, both sides null) are adjudication output, not A/B conflict.
            if asserted_a or asserted_b:
                predicate_disagreements[triple[1]] += 1
        union = agreed | only_a | only_b
        asserted_a = len(agreed) + len(only_a)
        asserted_b = len(agreed) + len(only_b)
        jaccard = len(agreed) / len(union) if union else 1.0
        is_exact = not only_a and not only_b
        samples_ok = row.get("source_annotation_status") or {}
        ready = (
            samples_ok.get("annotator_a") == "complete"
            and samples_ok.get("annotator_b") == "complete"
        )
        if ready:
            bucket["samples"] += 1
            bucket["exact"] += int(is_exact)
            bucket["jaccard_sum"] += jaccard
            bucket["asserted"] += len(union)
            bucket["asserted_a"] += asserted_a
            bucket["asserted_b"] += asserted_b
            bucket["only_a"] += len(only_a)
            bucket["only_b"] += len(only_b)
            bucket["agreed"] += len(agreed)
            exact += int(is_exact)
            jaccard_total += jaccard
            asserted += len(union)
            asserted_a_total += asserted_a
            asserted_b_total += asserted_b
            only_a_total += len(only_a)
            only_b_total += len(only_b)
            agreed_total += len(agreed)
    for bucket in per_task.values():
        n = bucket["samples"]
        bucket["sample_exact_agreement_rate"] = (bucket["exact"] / n) if n else None
        bucket["relation_set_jaccard_mean"] = (
            bucket["jaccard_sum"] / n if n else None
        )
        bucket["micro_relation_set_jaccard"] = (
            bucket["agreed"] / bucket["asserted"] if bucket["asserted"] else None
        )
        positive_denominator = bucket["asserted_a"] + bucket["asserted_b"]
        bucket["positive_agreement_rate"] = (
            2 * bucket["agreed"] / positive_denominator
            if positive_denominator
            else None
        )
    return {
        "n_samples_compared": samples,
        "sample_exact_agreement_rate": (exact / samples) if samples else None,
        "relation_set_jaccard_mean": (jaccard_total / samples) if samples else None,
        "micro_relation_set_jaccard": (agreed_total / asserted) if asserted else None,
        "positive_agreement_rate": (
            2 * agreed_total / (asserted_a_total + asserted_b_total)
            if asserted_a_total + asserted_b_total
            else None
        ),
        "asserted_relations_union": asserted,
        "annotator_a_relations": asserted_a_total,
        "annotator_b_relations": asserted_b_total,
        "agreed_relations": agreed_total,
        "annotator_a_only_relations": only_a_total,
        "annotator_b_only_relations": only_b_total,
        "predicate_disagreement_histogram": dict(
            predicate_disagreements.most_common()
        ),
        "per_task": per_task,
    }


def freeze_adjudication(
    *,
    manifest_path: Path,
    adjudication_path: Path,
    output_path: Path,
    annotation_source: str = "human",
) -> dict[str, Any]:
    expected_type = annotation_type_for_source(annotation_source)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(adjudication_path)
    expected = {row["sample_key"] for row in manifest["samples"]}
    by_key = {str(row.get("sample_key") or ""): row for row in rows}
    if set(by_key) != expected or len(by_key) != len(rows):
        raise ValueError("adjudication sample keys do not match the manifest exactly")
    frozen_samples: list[dict[str, Any]] = []
    annotator_ids: set[str] = set()
    adjudicator_ids: set[str] = set()
    annotator_models: set[str] = set()
    adjudicator_models: set[str] = set()
    for manifest_row in manifest["samples"]:
        key = manifest_row["sample_key"]
        row = by_key[key]
        adjudication = row.get("adjudication") or {}
        if adjudication.get("status") != "complete":
            raise ValueError(f"sample {key}: adjudication is not complete")
        adjudicator_id = str(adjudication.get("adjudicator_id") or "").strip()
        if not adjudicator_id:
            raise ValueError(f"sample {key}: adjudicator_id is required")
        if adjudication.get("adjudicator_type") != expected_type:
            raise ValueError(
                f"sample {key}: adjudicator_type must be {expected_type!r}"
            )
        if not str(adjudication.get("completed_at") or "").strip():
            raise ValueError(f"sample {key}: adjudication completed_at is required")
        if expected_type == "model":
            for field in ("adjudicator_model_id", "adjudicator_prompt_version"):
                if not str(adjudication.get(field) or "").strip():
                    raise ValueError(f"sample {key}: {field} is required")
            adjudicator_models.add(str(adjudication["adjudicator_model_id"]))
        provenance = row.get("source_annotation_provenance") or {}
        source_ids = {
            str((provenance.get(name) or {}).get("id") or "").strip()
            for name in ("annotator_a", "annotator_b")
        }
        if "" in source_ids or len(source_ids) != 2:
            raise ValueError(f"sample {key}: two distinct source annotators are required")
        if adjudicator_id in source_ids:
            raise ValueError(
                f"sample {key}: adjudicator must differ from both source annotators"
            )
        declared_types = {
            (provenance.get(name) or {}).get("type")
            for name in ("annotator_a", "annotator_b")
        }
        if declared_types != {expected_type}:
            raise ValueError(
                f"sample {key}: source annotators must declare "
                f"annotator_type={expected_type!r}"
            )
        for name in ("annotator_a", "annotator_b"):
            model_id = str((provenance.get(name) or {}).get("model_id") or "").strip()
            if expected_type == "model":
                if not model_id:
                    raise ValueError(f"sample {key}: source annotator {name} lacks model_id")
                annotator_models.add(model_id)
        annotator_ids.update(source_ids)
        adjudicator_ids.add(adjudicator_id)
        accepted: dict[tuple[str, str, str], dict[str, Any]] = {}
        if adjudication.get("accept_agreed_relations"):
            for agreement in row.get("agreed_relations") or []:
                triple = tuple(agreement.get("triple") or [])
                if len(triple) != 3:
                    raise ValueError(f"sample {key}: malformed agreed triple")
                evidence_source = agreement.get("annotator_a") or {}
                accepted[triple] = {
                    "triple": list(triple),
                    "evidence_mode": evidence_source.get("evidence_mode"),
                    "evidence": evidence_source.get("evidence") or [],
                    "decision_source": "double_annotation_agreement",
                }
        for disagreement in row.get("disagreements") or []:
            decision = disagreement.get("decision")
            if decision not in {"entailed", "not_entailed"}:
                raise ValueError(f"sample {key}: unresolved disagreement")
            if decision == "entailed":
                triple = tuple(disagreement.get("triple") or [])
                evidence = disagreement.get("adjudicated_evidence") or []
                mode = disagreement.get("adjudicated_evidence_mode")
                if len(triple) != 3 or len(evidence) not in {1, 2}:
                    raise ValueError(f"sample {key}: malformed adjudicated relation")
                if mode != ("explicit" if len(evidence) == 1 else "cross_sentence"):
                    raise ValueError(f"sample {key}: invalid adjudicated evidence mode")
                context = str((row.get("inputs") or {}).get("context") or "")
                if any(str(quote) not in context for quote in evidence):
                    raise ValueError(f"sample {key}: adjudicated evidence is not verbatim")
                accepted[triple] = {
                    "triple": list(triple),
                    "evidence_mode": mode,
                    "evidence": evidence,
                    "decision_source": "adjudication",
                }
        frozen_samples.append(
            {
                "sample_key": key,
                "task": manifest_row["task"],
                "source_index": manifest_row["source_index"],
                "relations": [accepted[key] for key in sorted(accepted)],
                "annotation_complete": True,
            }
        )
    base_meta: dict[str, Any] = {
        "protocol": MODEL_PROTOCOL if expected_type == "model" else PROTOCOL,
        "sample_protocol": PROTOCOL,
        "packet_id": manifest["meta"]["packet_id"],
        "manifest_sha256": _file_sha256(manifest_path),
        "adjudication_sha256": _file_sha256(adjudication_path),
        "n_samples": len(frozen_samples),
        "n_relations": sum(len(row["relations"]) for row in frozen_samples),
        "annotation_provenance": "declared_by_annotation_files_not_identity_verified",
        "source_annotator_ids": sorted(annotator_ids),
        "adjudicator_ids": sorted(adjudicator_ids),
    }
    if expected_type == "human":
        base_meta.update(
            {
                "annotation_source": "human_adjudicated",
                "declared_human_annotation": True,
                "human_annotated": True,
                "confirmatory_eligible": True,
                "calibration_ready": True,
            }
        )
    else:
        # Weak labels: usable for exploratory engineering calibration only. The
        # confirmatory gate stays closed until a real human audit exists, and
        # `calibration_ready` is deliberately False so that no downstream runner
        # can mistake this artifact for a deployment-grade label set.
        base_meta.update(
            {
                "annotation_source": MODEL_ANNOTATION_SOURCE_ID,
                "declared_human_annotation": False,
                "human_annotated": False,
                "confirmatory_eligible": False,
                "calibration_ready": False,
                "exploratory_calibration_ready": True,
                "calibration_scope": "exploratory_dual_llm_weak_label",
                "claim_label": MODEL_CLAIM_LABEL,
                "source_annotator_models": sorted(annotator_models),
                "adjudicator_models": sorted(adjudicator_models),
                "requires_human_audit_before_confirmatory": True,
                "human_audit_scope": (
                    "all_disagreements, all_positive_relations, "
                    "random_20pct_agreed_negatives"
                ),
            }
        )
    payload = {"meta": base_meta, "samples": frozen_samples}
    _atomic_write_json(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare.add_argument("--count-per-task", type=int, default=15)
    prepare.add_argument("--seed", type=int, default=20260923)
    prepare.add_argument("--max-context-chars", type=int, default=6000)
    prepare.add_argument("--exclude-prefix", type=int, default=30)

    prepare_model = subparsers.add_parser("prepare-model")
    prepare_model.add_argument(
        "--output-dir", type=Path, default=MODEL_DEFAULT_OUTPUT_DIR
    )
    prepare_model.add_argument("--source-packet-dir", type=Path, default=None)
    prepare_model.add_argument("--count-per-task", type=int, default=15)
    prepare_model.add_argument("--seed", type=int, default=20260923)
    prepare_model.add_argument("--max-context-chars", type=int, default=6000)
    prepare_model.add_argument("--exclude-prefix", type=int, default=30)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--annotations", type=Path, required=True)
    validate.add_argument(
        "--annotation-source", choices=ANNOTATION_SOURCES, default="human"
    )

    compare = subparsers.add_parser("compare")
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--annotations-a", type=Path, required=True)
    compare.add_argument("--annotations-b", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument(
        "--annotation-source", choices=ANNOTATION_SOURCES, default="human"
    )

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--adjudication", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument(
        "--annotation-source", choices=ANNOTATION_SOURCES, default="human"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        if args.count_per_task < 1:
            raise SystemExit("--count-per-task must be positive")
        manifest = prepare_packet(
            args.output_dir,
            count_per_task=args.count_per_task,
            seed=args.seed,
            max_context_chars=args.max_context_chars,
            prefix=args.exclude_prefix,
        )
        print(
            f"Prepared packet={manifest['meta']['packet_id']} "
            f"samples={manifest['meta']['n_samples']} at {args.output_dir}"
        )
        return 0
    if args.command == "prepare-model":
        if args.count_per_task < 1:
            raise SystemExit("--count-per-task must be positive")
        manifest = prepare_model_packet(
            args.output_dir,
            source_packet_dir=args.source_packet_dir,
            count_per_task=args.count_per_task,
            seed=args.seed,
            max_context_chars=args.max_context_chars,
            prefix=args.exclude_prefix,
        )
        print(
            f"Prepared model packet={manifest['meta']['packet_id']} "
            f"samples={manifest['meta']['n_samples']} at {args.output_dir}"
        )
        return 0
    if args.command == "validate":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        rows = validate_annotation_file(
            args.annotations, manifest, annotation_source=args.annotation_source
        )
        complete = sum(row["annotation"]["status"] == "complete" for row in rows)
        print(
            f"Valid annotations: source={args.annotation_source} "
            f"rows={len(rows)} complete={complete}"
        )
        return 0
    if args.command == "compare":
        summary = compare_annotations(
            manifest_path=args.manifest,
            annotation_a_path=args.annotations_a,
            annotation_b_path=args.annotations_b,
            output_path=args.output,
            annotation_source=args.annotation_source,
            agreement_path=args.output.with_name("model_agreement.json")
            if args.annotation_source == "model"
            else args.output.with_name("agreement.json"),
        )
        print(
            "Compared annotations: "
            f"source={args.annotation_source} "
            f"disagreements={summary.get('disagreements')} "
            f"exact_agreement={summary.get('sample_exact_agreement_rate')} "
            f"jaccard={summary.get('relation_set_jaccard_mean')} "
            f"positive_agreement={summary.get('positive_agreement_rate')}"
        )
        return 0
    payload = freeze_adjudication(
        manifest_path=args.manifest,
        adjudication_path=args.adjudication,
        output_path=args.output,
        annotation_source=args.annotation_source,
    )
    meta = payload["meta"]
    print(
        f"Frozen annotations: source={meta['annotation_source']} "
        f"samples={meta['n_samples']} relations={meta['n_relations']} "
        f"human_annotated={meta['human_annotated']} "
        f"confirmatory_eligible={meta.get('confirmatory_eligible')} "
        f"calibration_ready={meta['calibration_ready']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
