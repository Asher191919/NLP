"""Build a deterministic, relation-rich synthetic holdout with nonce entities."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import run_official_blind_eval as blind


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "holdouts" / "nonce_relation_v1"
PROTOCOL = "nonce_relation_holdout_v1"
DEFAULT_SEED = 20260926
DEFAULT_SAMPLES = 30
PREDICATES = (
    "part_of",
    "has part",
    "develops_from",
    "derives from",
    "located in",
    "regulates",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        + "\n"
    ).encode("utf-8")


def entity_name(seed: int, sample_index: int, role: int) -> str:
    digest = hashlib.sha256(
        f"{PROTOCOL}|{seed}|{sample_index}|{role}".encode("ascii")
    ).hexdigest()[:10]
    return f"entity_{digest}"


def _explicit_sentence(
    subject: str, predicate: str, obj: str, surface_form: str
) -> str:
    if surface_form == "canonical":
        templates = {
            "part_of": "{s} has the ontology relation part_of to {o}.",
            "has part": "{s} has part {o}.",
            "develops_from": "{s} develops_from {o}.",
            "derives from": "{s} derives from {o}.",
            "located in": "{s} is located in {o}.",
            "regulates": "{s} regulates {o}.",
        }
    else:
        templates = {
            "part_of": "{o} includes {s} as a component.",
            "has part": "{o} is a component contained by {s}.",
            "develops_from": "{o} is the developmental precursor of {s}.",
            "derives from": "{o} is the source material for {s}.",
            "located in": "{o} houses {s}.",
            "regulates": "{s} controls the activity of {o}.",
        }
    return templates[predicate].format(s=subject, o=obj)


def _cross_sentences(
    subject: str, predicate: str, obj: str, surface_form: str
) -> list[str]:
    first = f"The focal item in this statement is {subject}."
    if surface_form == "canonical":
        templates = {
            "part_of": "It has the ontology relation part_of to {o}.",
            "has part": "It has part {o}.",
            "develops_from": "It develops_from {o}.",
            "derives from": "It derives from {o}.",
            "located in": "It is located in {o}.",
            "regulates": "It regulates {o}.",
        }
    else:
        templates = {
            "part_of": "It belongs within {o} as a component.",
            "has part": "One of its components is {o}.",
            "develops_from": "Its developmental precursor is {o}.",
            "derives from": "Its source material is {o}.",
            "located in": "Its containing site is {o}.",
            "regulates": "The activity it controls belongs to {o}.",
        }
    return [first, templates[predicate].format(o=obj)]


def build_sample(sample_index: int, *, seed: int) -> dict[str, Any]:
    sample_id = f"nonce_rel_{sample_index:03d}"
    surface_form = "canonical" if sample_index < DEFAULT_SAMPLES // 2 else "paraphrased"
    entities = [entity_name(seed, sample_index, role) for role in range(10)]
    annotations = []
    sentences = [
        f"Controlled fictional relation record {sample_id}; use only statements in this record."
    ]
    for slot in range(4):
        predicate = PREDICATES[(sample_index * 4 + slot) % len(PREDICATES)]
        subject, obj = entities[slot * 2], entities[slot * 2 + 1]
        if slot < 2:
            evidence_mode = "explicit"
            evidence = [_explicit_sentence(subject, predicate, obj, surface_form)]
        else:
            evidence_mode = "cross_sentence"
            evidence = _cross_sentences(subject, predicate, obj, surface_form)
        sentences.extend(evidence)
        annotations.append(
            {
                "triple": [subject, predicate, obj],
                "evidence_mode": evidence_mode,
                "evidence": evidence,
            }
        )

    negative_subject, negative_object = entities[8], entities[9]
    sentences.append(
        f"{negative_subject} and {negative_object} were inspected together, but "
        f"{negative_subject} does not regulate {negative_object} and does not have "
        f"the ontology relation part_of to {negative_object}."
    )
    controls = [
        [negative_subject, "regulates", negative_object],
        [negative_subject, "part_of", negative_object],
    ]
    return {
        "id": sample_id,
        "sample_index": sample_index,
        "surface_form": surface_form,
        "context": " ".join(sentences),
        "gold_relations": [row["triple"] for row in annotations],
        "gold_relation_annotations": annotations,
        "hard_negative_controls": controls,
    }


def official_overlap_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    official_samples = {
        task: [sample for _, sample in blind.load_samples(task, 0, 0)]
        for task in ("a", "b")
    }
    official_text = json.dumps(official_samples, ensure_ascii=False).lower()
    official_gold = set()
    for task, rows in official_samples.items():
        gold_key = blind.TASK_CONFIG[task]["gold_key"]
        for row in rows:
            official_gold.update(
                tuple(triple)
                for triple in blind.normalize_triples(row.get(gold_key) or [])
            )
    entities = {
        endpoint
        for sample in samples
        for triple in sample["gold_relations"] + sample["hard_negative_controls"]
        for endpoint in (triple[0], triple[2])
    }
    gold = {
        tuple(triple) for sample in samples for triple in sample["gold_relations"]
    }
    return {
        "official_2026_entity_string_overlap_count": sum(
            entity.lower() in official_text for entity in entities
        ),
        "official_2026_exact_gold_triple_overlap_count": len(gold & official_gold),
        "official_2026_sources": {
            task: {
                "path": str(blind.TASK_CONFIG[task]["path"].resolve()),
                "sha256": file_sha256(blind.TASK_CONFIG[task]["path"]),
            }
            for task in ("a", "b")
        },
    }


def audit_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) != DEFAULT_SAMPLES:
        raise ValueError(f"expected {DEFAULT_SAMPLES} samples")
    sample_ids = [sample["id"] for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample ids must be unique")

    all_gold = []
    all_controls = []
    all_entities = []
    predicate_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    surface_counts: Counter[str] = Counter()
    for sample in samples:
        if len(sample["gold_relations"]) != 4:
            raise ValueError("every sample must contain four gold relations")
        if len(sample["hard_negative_controls"]) != 2:
            raise ValueError("every sample must contain two hard-negative controls")
        if sample["gold_relations"] != [
            row["triple"] for row in sample["gold_relation_annotations"]
        ]:
            raise ValueError("gold relation annotations drifted")
        context = sample["context"]
        for row in sample["gold_relation_annotations"]:
            subject, predicate, obj = row["triple"]
            if subject not in context or obj not in context:
                raise ValueError("gold endpoint missing from context")
            if not all(span in context for span in row["evidence"]):
                raise ValueError("gold evidence must be verbatim")
            predicate_counts[predicate] += 1
            evidence_counts[row["evidence_mode"]] += 1
            all_gold.append(tuple(row["triple"]))
            all_entities.extend((subject, obj))
        for control in sample["hard_negative_controls"]:
            if control[0] not in context or control[2] not in context:
                raise ValueError("hard-negative endpoint missing from context")
            all_controls.append(tuple(control))
            all_entities.extend((control[0], control[2]))
        surface_counts[sample["surface_form"]] += 1

    if len(all_gold) != len(set(all_gold)):
        raise ValueError("gold triples must be unique")
    if set(all_gold) & set(all_controls):
        raise ValueError("hard-negative controls must not be gold")
    if len(set(all_entities)) != DEFAULT_SAMPLES * 10:
        raise ValueError("nonce entities must be unique across samples")
    expected_predicates = {predicate: 20 for predicate in PREDICATES}
    if dict(predicate_counts) != expected_predicates:
        raise ValueError("predicate support is not balanced")
    if dict(evidence_counts) != {"explicit": 60, "cross_sentence": 60}:
        raise ValueError("evidence modes are not balanced")
    if dict(surface_counts) != {"canonical": 15, "paraphrased": 15}:
        raise ValueError("surface forms are not balanced")
    return {
        "samples": len(samples),
        "gold_relations": len(all_gold),
        "hard_negative_controls": len(all_controls),
        "unique_entities": len(set(all_entities)),
        "predicate_counts": dict(sorted(predicate_counts.items())),
        "evidence_mode_counts": dict(sorted(evidence_counts.items())),
        "surface_form_counts": dict(sorted(surface_counts.items())),
    }


def build_package(seed: int) -> tuple[dict[str, Any], bytes, bytes, dict[str, Any]]:
    samples = [build_sample(index, seed=seed) for index in range(DEFAULT_SAMPLES)]
    balance = audit_samples(samples)
    overlap = official_overlap_audit(samples)
    if overlap["official_2026_entity_string_overlap_count"] != 0:
        raise ValueError("nonce entities overlap official 2026 data")
    if overlap["official_2026_exact_gold_triple_overlap_count"] != 0:
        raise ValueError("synthetic gold overlaps official 2026 gold")
    dataset = {
        "meta": {
            "protocol": PROTOCOL,
            "seed": seed,
            "synthetic": True,
            "human_annotated": False,
            "contains_protected_gold": True,
            "created_for": "low_exposure_text_grounded_nontax_evaluation",
            "primary_metric": "exact_nontax_micro_f1",
        },
        "samples": samples,
    }
    dataset_bytes = _json_bytes(dataset)
    model_inputs = [{"id": row["id"], "context": row["context"]} for row in samples]
    inputs_bytes = _jsonl_bytes(model_inputs)
    package_id = _sha256_bytes(dataset_bytes)[:20]
    manifest = {
        "meta": {
            "protocol": PROTOCOL,
            "package_id": package_id,
            "seed": seed,
            "status": "frozen",
            "contains_source_text": False,
            "contains_protected_gold": False,
            "model_input_fields": ["id", "context"],
            "confirmatory_eligible": False,
            "scientific_deployable": False,
        },
        "files": {
            "dataset.json": {"sha256": _sha256_bytes(dataset_bytes)},
            "model_inputs.jsonl": {"sha256": _sha256_bytes(inputs_bytes)},
        },
        "balance_audit": balance,
        "official_overlap_audit": overlap,
        "locked_protocol": {
            "sample_count": DEFAULT_SAMPLES,
            "gold_relations_per_sample": 4,
            "hard_negative_controls_per_sample": 2,
            "predicate_inventory": list(PREDICATES),
            "surface_strata": ["canonical", "paraphrased"],
            "evidence_strata": ["explicit", "cross_sentence"],
            "gold_may_not_enter_prompts": True,
            "calibrator_may_not_be_fit_or_applied": True,
            "allow_generation_tuning": False,
            "allow_deployment_change": False,
            "interpretation": "synthetic_low_exposure_holdout_not_naturalistic_benchmark",
        },
        "samples": [
            {
                "sample_index": row["sample_index"],
                "sample_id": row["id"],
                "surface_form": row["surface_form"],
                "context_sha256": _sha256_bytes(row["context"].encode("utf-8")),
                "gold_relation_count": len(row["gold_relations"]),
                "hard_negative_count": len(row["hard_negative_controls"]),
            }
            for row in samples
        ],
    }
    return dataset, dataset_bytes, inputs_bytes, manifest


def preserve_bytes(path: Path, payload: bytes) -> bool:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"frozen holdout differs: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--audit-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, dataset_bytes, inputs_bytes, manifest = build_package(args.seed)
    output_dir = Path(args.output_dir)
    manifest_bytes = _json_bytes(manifest)
    if args.audit_only:
        existing = output_dir / "manifest.json"
        if existing.exists() and existing.read_bytes() != manifest_bytes:
            raise SystemExit("existing frozen holdout manifest differs")
        print(
            f"holdout samples={manifest['balance_audit']['samples']} "
            f"gold={manifest['balance_audit']['gold_relations']} "
            f"controls={manifest['balance_audit']['hard_negative_controls']} "
            "official_entity_overlap=0 official_triple_overlap=0",
            flush=True,
        )
        print("Audit only; no files changed and no API calls made", flush=True)
        return 0
    created = {
        "dataset.json": preserve_bytes(output_dir / "dataset.json", dataset_bytes),
        "model_inputs.jsonl": preserve_bytes(
            output_dir / "model_inputs.jsonl", inputs_bytes
        ),
        "manifest.json": preserve_bytes(output_dir / "manifest.json", manifest_bytes),
    }
    action = "Created" if any(created.values()) else "Verified"
    print(
        f"{action} {output_dir} package_id={manifest['meta']['package_id']} "
        f"gold={manifest['balance_audit']['gold_relations']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
