"""Build and run a cross-year, relation-only calibration set.

The source contexts are public LLMs4OL 2025 Text2Onto training documents. Gold
relations come from the 2025 Task D training pairs and are joined locally by
endpoint co-occurrence. Any document containing a gold triple found anywhere in
the LLMs4OL 2026 Task A/B training data is removed as a whole. Gold is never
serialized into a model prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import run_official_blind_eval as blind
import run_relation_v2_offline as offline
from ontobridg.relations.calibration import BetaIsotonicCalibrator, calibration_metrics
from ontobridg.relations.canonicalization import (
    build_entity_inventory,
    canonicalize_predicate,
    canonicalize_triple,
)


ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_2025 = ROOT / "data" / "LLMs4OL-Challenge" / "2025"
PROTOCOL = "llms4ol2025_cross_year_relation_calibration_v1"
SYSTEM = "open_vocab_guarded_v5_axioms_only"
ALLOWED_NONTAX_PREDICATES = "equivalent class,disjoint with"
TEXT_DOMAINS = ("ecology", "engineering", "scholarly")
RELATION_DOMAINS = {
    "FoodOn": "foodon",
    "GO": "go",
    "SWEET": "sweet",
}
SYMMETRIC_PREDICATES = {"disjoint with", "equivalent class"}


def _rank(seed: int, *parts: str) -> str:
    value = ":".join((str(seed), *parts))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_triple(values: list[Any]) -> tuple[str, str, str]:
    subject = blind._norm(values[0])
    predicate, reverse, _ = canonicalize_predicate(values[1])
    obj = blind._norm(values[2])
    if reverse:
        subject, obj = obj, subject
    return subject, predicate, obj


def _overlap_key(values: tuple[str, str, str]) -> tuple[str, str, str]:
    subject, predicate, obj = values
    if predicate in SYMMETRIC_PREDICATES and obj < subject:
        subject, obj = obj, subject
    return subject, predicate, obj


def load_2026_train_gold() -> set[tuple[str, str, str]]:
    triples: set[tuple[str, str, str]] = set()
    for task in ("a", "b"):
        gold_key = blind.TASK_CONFIG[task]["gold_key"]
        for _, sample in blind.load_samples(task, 0, 0):
            triples.update(
                _overlap_key(_canonical_triple(list(triple)))
                for triple in sample.get(gold_key) or []
                if isinstance(triple, (list, tuple)) and len(triple) >= 3
            )
    return triples


def load_2026_train_contexts() -> set[str]:
    return {
        blind._compact(sample.get("context") or "")
        for task in ("a", "b")
        for _, sample in blind.load_samples(task, 0, 0)
    }


def load_cross_year_sources() -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[str, dict[str, list[str]]],
    dict[str, list[dict[str, Any]]],
    list[Path],
]:
    documents: dict[tuple[str, str], dict[str, Any]] = {}
    term_docs: dict[str, dict[str, list[str]]] = {}
    source_paths: list[Path] = []
    for domain in TEXT_DOMAINS:
        train = CHALLENGE_2025 / "TaskA-Text2Onto" / domain / "train"
        documents_path = train / "documents.jsonl"
        term_docs_path = train / "terms2docs.json"
        source_paths.extend((documents_path, term_docs_path))
        with documents_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                documents[(domain, str(row["id"]))] = row
        mapping = json.loads(term_docs_path.read_text(encoding="utf-8"))
        term_docs[domain] = {
            blind._norm(term): [str(doc_id) for doc_id in doc_ids]
            for term, doc_ids in mapping.items()
        }

    relation_pairs: dict[str, list[dict[str, Any]]] = {}
    for domain, prefix in RELATION_DOMAINS.items():
        path = (
            CHALLENGE_2025
            / "TaskD-NonTaxonomicRE"
            / domain
            / "train"
            / f"{prefix}_train_re_pairs.json"
        )
        source_paths.append(path)
        relation_pairs[domain] = json.loads(path.read_text(encoding="utf-8"))
    return documents, term_docs, relation_pairs, source_paths


def build_cross_year_samples(
    *, seed: int
) -> tuple[list[tuple[int, dict[str, Any]]], dict[int, dict[str, Any]], dict[str, Any]]:
    documents, term_docs, relation_pairs, source_paths = load_cross_year_sources()
    triples_by_document: dict[
        tuple[str, str], dict[tuple[str, str, str], set[str]]
    ] = defaultdict(lambda: defaultdict(set))
    linked_occurrences = 0
    for relation_domain, pairs in relation_pairs.items():
        for pair in pairs:
            head = blind._norm(pair.get("head") or "")
            tail = blind._norm(pair.get("tail") or "")
            if not head or not tail:
                continue
            triple = _canonical_triple([head, pair.get("relation") or "", tail])
            for text_domain, mapping in term_docs.items():
                if head not in mapping or tail not in mapping:
                    continue
                common_docs = sorted(set(mapping[head]) & set(mapping[tail]))
                for doc_id in common_docs:
                    document = documents[(text_domain, doc_id)]
                    context = f"Title: {document['title']}\n\nContent:\n{document['text']}"
                    compact_context = blind._compact(context)
                    if (
                        blind._compact(head) not in compact_context
                        or blind._compact(tail) not in compact_context
                    ):
                        continue
                    linked_occurrences += 1
                    triples_by_document[(text_domain, doc_id)][triple].add(
                        relation_domain
                    )

    excluded_gold = load_2026_train_gold()
    excluded_contexts = load_2026_train_contexts()
    blocked: dict[tuple[str, str], list[str]] = {}
    eligible: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    for key, triple_sources in triples_by_document.items():
        document = documents[key]
        context = f"Title: {document['title']}\n\nContent:\n{document['text']}"
        reasons = []
        if any(_overlap_key(triple) in excluded_gold for triple in triple_sources):
            reasons.append("contains_2026_train_gold_triple")
        if blind._compact(context) in excluded_contexts:
            reasons.append("exact_2026_train_context")
        if reasons:
            blocked[key] = reasons
        else:
            eligible[key] = set(triple_sources)

    all_eligible_triples = set().union(*eligible.values()) if eligible else set()
    uncovered = set(all_eligible_triples)
    covered: set[tuple[str, str, str]] = set()
    selected_keys: list[tuple[str, str]] = []
    while uncovered:
        choices = []
        for key, triples in eligible.items():
            if key in selected_keys or triples & covered:
                continue
            gain = triples & uncovered
            if gain:
                choices.append(
                    (-len(gain), _rank(seed, key[0], key[1]), key)
                )
        if not choices:
            raise ValueError(
                f"could not select duplicate-free documents for {len(uncovered)} triples"
            )
        _, _, chosen = min(choices)
        selected_keys.append(chosen)
        covered.update(eligible[chosen])
        uncovered.difference_update(eligible[chosen])

    selected_keys.sort()
    selected: list[tuple[int, dict[str, Any]]] = []
    metadata: dict[int, dict[str, Any]] = {}
    selected_triples: set[tuple[str, str, str]] = set()
    for index, key in enumerate(selected_keys):
        text_domain, doc_id = key
        document = documents[key]
        triples = sorted(eligible[key])
        overlap = selected_triples & set(triples)
        if overlap:
            raise ValueError(f"duplicate selected gold triples: {sorted(overlap)}")
        selected_triples.update(triples)
        context = f"Title: {document['title']}\n\nContent:\n{document['text']}"
        sample = {
            "id": f"llms4ol2025:{text_domain}:{doc_id}",
            "context": context,
            "primitive-ontology-triples": [list(triple) for triple in triples],
        }
        selected.append((index, sample))
        relation_domains = sorted(
            {
                source
                for triple in triples
                for source in triples_by_document[key][triple]
            }
        )
        metadata[index] = {
            "source_year": 2025,
            "text_domain": text_domain,
            "source_document_id_sha256": blind._sha256(doc_id),
            "relation_domains": relation_domains,
            "visible_gold": len(triples),
            "visible_gold_predicates": sorted({triple[1] for triple in triples}),
            "selection_purpose": "cross_year_relation_calibration",
        }

    summary = {
        "source_repository_license": "MIT",
        "underlying_ontology_license_reverified": False,
        "source_files": [
            {"path": str(path.resolve()), "sha256": _file_sha256(path)}
            for path in source_paths
        ],
        "relation_pairs_total": sum(len(rows) for rows in relation_pairs.values()),
        "endpoint_linked_occurrences": linked_occurrences,
        "endpoint_linked_documents": len(triples_by_document),
        "blocked_documents": len(blocked),
        "blocked_reason_counts": dict(
            Counter(reason for reasons in blocked.values() for reason in reasons)
        ),
        "eligible_documents": len(eligible),
        "eligible_unique_gold_triples": len(all_eligible_triples),
        "selected_documents": len(selected),
        "selected_unique_gold_triples": len(selected_triples),
        "selected_predicate_counts": dict(
            Counter(triple[1] for triple in selected_triples)
        ),
        "gold_triple_overlap_with_2026_train": len(
            {_overlap_key(triple) for triple in selected_triples} & excluded_gold
        ),
        "duplicate_gold_across_selected_documents": False,
        "selection_seed": seed,
    }
    return selected, metadata, summary


def build_audit_manifest(args: argparse.Namespace) -> dict[str, Any]:
    selected, metadata, pool = build_cross_year_samples(seed=args.seed)
    rows = []
    generation_hits = 0
    exact_hits = 0
    prompt_chars = 0
    for index, sample in selected:
        prompt = blind.build_prompt(
            sample,
            "a",
            SYSTEM,
            args.max_context_chars,
            args.confidence_threshold,
            args.max_nontax,
            args.implicit_confidence_threshold,
        )
        prompt_hash = blind._sha256(prompt)
        postprocess_hash = blind.postprocess_sha256(
            confidence_threshold=args.confidence_threshold,
            implicit_confidence_threshold=args.implicit_confidence_threshold,
            calibrated_confidence_threshold=args.calibrated_confidence_threshold,
            max_nontax=args.max_nontax,
            max_implicit_gap_chars=args.max_implicit_gap_chars,
            calibrator=None,
            allowed_nontax_predicates=args.allowed_nontax_predicates,
        )
        sample_id = str(sample["id"])
        cache = blind.cache_path(args.model, "a", SYSTEM, index, sample_id)
        generation_hit = blind.cache_has_reusable_generation(
            cache, prompt_sha256=prompt_hash
        )
        exact_hit = blind.cache_is_replayable(
            cache,
            system=SYSTEM,
            prompt_sha256=prompt_hash,
            postprocess_hash=postprocess_hash,
        )
        generation_hits += int(generation_hit)
        exact_hits += int(exact_hit)
        prompt_chars += len(prompt)
        rows.append(
            {
                "sample_index": index,
                "sample_id_sha256": blind._sha256(sample_id),
                **metadata[index],
                "input_fields": ["context"],
                "protected_gold_field": "primitive-ontology-triples",
                "context_chars_sent": min(
                    len(str(sample.get("context") or "")), args.max_context_chars
                ),
                "prompt": {
                    "sha256": prompt_hash,
                    "chars": len(prompt),
                    "generation_cache_hit": generation_hit,
                    "exact_cache_hit": exact_hit,
                },
            }
        )
    return {
        "meta": {
            "protocol": PROTOCOL,
            "audit_only": True,
            "selection_uses_gold_locally": True,
            "selection_is_calibration_only_not_unbiased_evaluation": True,
            "gold_visible_to_model": False,
            "contains_source_text": False,
            "external_destination": (
                args.base_url
                or os.environ.get("ONTOBRIDG_BASE_URL")
                or "https://api.openai.com/v1"
            ),
            "model": args.model,
            "system": SYSTEM,
            "seed": args.seed,
            "estimated_calls": len(selected),
            "estimated_generation_cache_hits": generation_hits,
            "estimated_exact_cache_replays": exact_hits,
            "estimated_local_reprocesses": generation_hits - exact_hits,
            "estimated_external_calls": len(selected) - generation_hits,
            "total_prompt_chars": prompt_chars,
            "disjoint_from_all_2026_train_gold": (
                pool["gold_triple_overlap_with_2026_train"] == 0
            ),
        },
        "pool": pool,
        "samples": rows,
    }


def collect_candidates(
    selected: list[tuple[int, dict[str, Any]]], payload: dict[str, Any]
) -> list[dict[str, Any]]:
    samples = dict(selected)
    candidates: list[dict[str, Any]] = []
    for record in payload["samples"]:
        if record.get("status") != "ok":
            continue
        index = int(record["sample_index"])
        sample = samples[index]
        context = str(sample.get("context") or "")
        gold = blind.categorize(
            blind.normalize_triples(sample.get("primitive-ontology-triples") or [])
        )["nontax"]
        symmetric_gold = {
            offline.relation_match_key(row, symmetric_order_invariant=True)
            for row in gold
        }
        cached = json.loads(Path(record["cache"]).read_text(encoding="utf-8"))
        guard = cached.get("guard_diagnostics") or {}
        diagnostics_by_position = {
            int(row["position"]): row
            for key in ("accepted_nontax_candidates", "rejected_nontax_candidates")
            for row in guard.get(key, [])
            if row.get("position") is not None
        }
        accepted_positions = {
            int(row["position"])
            for row in guard.get("accepted_nontax_candidates", [])
            if row.get("position") is not None
        }
        rejected = {
            int(row["position"]): list(row.get("reasons") or [])
            for row in guard.get("rejected_nontax_candidates", [])
            if row.get("position") is not None
        }
        inventory = build_entity_inventory(sample, task="a")
        for position, item in enumerate(blind.parse_model_items(cached["raw_text"])):
            if not isinstance(item, dict):
                continue
            normalized = blind.normalize_triples([item])
            if not normalized:
                continue
            canonical, trace = canonicalize_triple(
                normalized[0], context=context, entity_inventory=inventory
            )
            if (
                canonical[1] in blind.INSTANCE_RELATIONS
                or canonical[1] in blind.TAXONOMY_RELATIONS
            ):
                continue
            try:
                confidence = float(item.get("confidence"))
            except (TypeError, ValueError):
                continue
            candidates.append(
                {
                    "task": "cross_year",
                    "sample_index": index,
                    "stratum": None,
                    "split": "calibration",
                    "confidence": confidence,
                    "position": position,
                    "raw_triple": normalized[0],
                    "canonical_triple": canonical,
                    "raw_correct": tuple(normalized[0]) in gold,
                    "canonical_correct": tuple(canonical) in gold,
                    "symmetric_semantic_correct": offline.relation_match_key(
                        canonical, symmetric_order_invariant=True
                    )
                    in symmetric_gold,
                    "canonicalization_changed": bool(trace["changed"]),
                    "near_match_type": offline.classify_near_match(canonical, gold)[0],
                    "near_gold_triples": offline.classify_near_match(canonical, gold)[1],
                    "evidence_mode": blind._norm(item.get("evidence_mode") or ""),
                    "evidence_gap_chars": diagnostics_by_position.get(
                        position, {}
                    ).get("evidence_gap_chars"),
                    "postprocess_status": (
                        "accepted"
                        if position in accepted_positions
                        else "rejected"
                        if position in rejected
                        else "not_recorded"
                    ),
                    "rejection_reasons": rejected.get(position, []),
                }
            )
    return candidates


def build_offline_report(
    candidates: list[dict[str, Any]], selected: list[tuple[int, dict[str, Any]]]
) -> dict[str, Any]:
    eligible = offline.deployment_eligible_candidates(candidates)
    calibrator = BetaIsotonicCalibrator().fit(
        [row["confidence"] for row in eligible],
        [row["canonical_correct"] for row in eligible],
    )
    positive_samples = len(
        {
            row["sample_index"]
            for row in eligible
            if row["canonical_correct"]
        }
    )
    deployable = (
        calibrator.deployable
        and positive_samples >= offline.MIN_DEPLOYMENT_POSITIVE_SAMPLES
    )
    calibrated = [calibrator.calibrate(row["confidence"]) for row in eligible]
    predicted_by_sample: dict[
        tuple[str, int], set[tuple[str, str, str]]
    ] = {}
    ranked_predictions_by_sample: dict[
        tuple[str, int], list[tuple[str, str, str]]
    ] = {}
    for row in candidates:
        if row["postprocess_status"] == "accepted":
            key = ("cross_year", int(row["sample_index"]))
            triple = tuple(row["canonical_triple"])
            predicted_by_sample.setdefault(key, set()).add(triple)
            ranked_predictions_by_sample.setdefault(key, []).append(triple)
    gold_by_sample = {
        ("cross_year", int(index)): set(
            blind.categorize(
                blind.normalize_triples(
                    sample.get("primitive-ontology-triples") or []
                )
            )["nontax"]
        )
        for index, sample in selected
    }
    return {
        "candidate_summary": {
            "all": offline.summarize_candidates(candidates),
            "deployment_population": offline.summarize_candidates(eligible),
        },
        "calibrator": calibrator.to_dict(),
        "deployment_support": {
            "deployable": deployable,
            "candidate_level_deployable": calibrator.deployable,
            "status": (
                "ok"
                if deployable
                else "insufficient_independent_positive_samples"
                if calibrator.deployable
                else calibrator.status
            ),
            "min_positive_samples": offline.MIN_DEPLOYMENT_POSITIVE_SAMPLES,
            "positive_candidates": sum(
                bool(row["canonical_correct"]) for row in eligible
            ),
            "positive_samples": positive_samples,
        },
        "calibration_metrics": {
            "raw": calibration_metrics(
                [row["confidence"] for row in eligible],
                [row["canonical_correct"] for row in eligible],
            ),
            "calibrated": calibration_metrics(
                calibrated,
                [row["canonical_correct"] for row in eligible],
            ),
        },
        "threshold_selection": offline.calibration_threshold_sweep(
            candidates, calibrator
        ),
        "predicate_policy": offline.calibration_predicate_policy(candidates),
        "symmetric_serialization_audit": offline.symmetric_serialization_report(
            predicted_by_sample,
            gold_by_sample,
            scope="accepted_candidates_only_cross_year_endpoint_linked_gold",
            ranked_predictions_by_sample=ranked_predictions_by_sample,
            max_per_sample=8,
        ),
        "exact_positive_candidates": [
            row for row in candidates if row["canonical_correct"]
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(allowed_nontax_predicates=ALLOWED_NONTAX_PREDICATES)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument(
        "--backend", choices=["openai", "openai_compat", "api"], default="openai"
    )
    parser.add_argument(
        "--model", default=os.environ.get("ONTOBRIDG_MODEL", "gpt-4o-mini")
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument(
        "--confidence-threshold", type=float, default=blind.DEFAULT_CONFIDENCE_THRESHOLD
    )
    parser.add_argument(
        "--implicit-confidence-threshold",
        type=float,
        default=blind.DEFAULT_IMPLICIT_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument(
        "--calibrated-confidence-threshold",
        type=float,
        default=blind.DEFAULT_CALIBRATED_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument("--max-nontax", type=int, default=blind.DEFAULT_MAX_NONTAX)
    parser.add_argument(
        "--max-implicit-gap-chars",
        type=int,
        default=blind.DEFAULT_MAX_IMPLICIT_GAP_CHARS,
    )
    parser.add_argument("--prior-failed-connection-attempts", type=int, default=0)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1 or args.workers > 4:
        raise SystemExit("--workers must be between 1 and 4")
    if args.prior_failed_connection_attempts < 0:
        raise SystemExit("--prior-failed-connection-attempts must be non-negative")
    args.calibration_file = None
    args.offset = 0
    selected, metadata, pool = build_cross_year_samples(seed=args.seed)
    args.limit = len(selected)
    tag = f"n{len(selected)}_seed{args.seed}"

    if args.audit_only:
        manifest = build_audit_manifest(args)
        output = (
            blind.RESULTS
            / f"cross_year_relation_calibration_audit_{SYSTEM}_{tag}.json"
        )
        blind.atomic_write_json(output, manifest)
        print(
            f"Audit only: samples={manifest['meta']['estimated_calls']} "
            f"external_calls={manifest['meta']['estimated_external_calls']} "
            f"gold={manifest['pool']['selected_unique_gold_triples']}",
            flush=True,
        )
        print(f"Wrote {output}", flush=True)
        return 0

    checkpoint = (
        blind.RESULTS
        / "checkpoints"
        / f"cross_year_relation_calibration_{blind._slug(args.backend)}_{SYSTEM}_{tag}.json"
    )
    payload = blind.run_system(
        args=args,
        task="a",
        system=SYSTEM,
        indexed_samples=selected,
        checkpoint_path=checkpoint,
    )
    for record in payload["samples"]:
        record.update(metadata[int(record["sample_index"])])
    payload["meta"].update(
        {
            "protocol": PROTOCOL,
            "selection": "cross_year_relation_calibration",
            "pool": pool,
        }
    )
    blind.atomic_write_json(checkpoint, payload)
    candidates = collect_candidates(selected, payload)
    offline_report = build_offline_report(candidates, selected)
    summary = payload["summary"]
    successful = int(summary["n_success"])
    failed = int(summary["n_failed"])
    runner_attempts = int(summary["successful_runner_attempts_total"])
    result = {
        "meta": {
            "protocol": PROTOCOL,
            "model": args.model,
            "system": SYSTEM,
            "seed": args.seed,
            "selection_uses_gold_locally": True,
            "selection_is_calibration_only_not_unbiased_evaluation": True,
            "gold_visible_to_model": False,
            "pool": pool,
            "attempt_ledger": {
                "planned_units": len(selected),
                "successful_units": successful,
                "failed_final_units": failed,
                "prior_failed_connection_attempts": (
                    args.prior_failed_connection_attempts
                ),
                "unit_attempts_including_prior_failures": (
                    successful + failed + args.prior_failed_connection_attempts
                ),
                "successful_runner_attempts_including_retries": runner_attempts,
                "low_level_attempts_including_prior_failures": (
                    runner_attempts + args.prior_failed_connection_attempts
                ),
            },
        },
        "generation": payload,
        "offline_report": offline_report,
    }
    output = blind.RESULTS / f"cross_year_relation_calibration_{SYSTEM}_{tag}.json"
    blind.atomic_write_json(output, result)
    print(
        f"Cross-year: ok={successful}/{len(selected)} "
        f"candidates={offline_report['candidate_summary']['all']['n']} "
        f"positive_samples={offline_report['deployment_support']['positive_samples']} "
        f"deployable={offline_report['deployment_support']['deployable']}",
        flush=True,
    )
    print(f"Wrote {output}", flush=True)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
