"""Leakage-controlled, resumable evaluation on LLMs4OL 2026 train data.

Task A model input: context only.
Task B model input: context, initial ontology, and officially provided terms/types.
Gold target triples are used only after prediction for scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ontobridg.eval.official_graph_similarity import exact_match
from ontobridg.llm.factory import build_llm
from ontobridg.relations.calibration import BetaIsotonicCalibrator
from ontobridg.relations.canonicalization import (
    CANONICALIZATION_VERSION,
    build_entity_inventory,
    canonicalize_predicate,
    canonicalize_triple,
)
from ontobridg.relations.predicate_inventory import (
    STYLE_OR,
    STYLE_SEMICOLON,
    inventory_labels,
    render_labels,
    route_for_system,
)

CHALLENGE = ROOT / "data" / "LLMs4OL-Challenge" / "2026"
RESULTS = ROOT / "experiments" / "results"
PROTOCOL = "llms4ol2026_blind_v1"
SYSTEMS = (
    "closed_vocab",
    "open_vocab",
    "open_vocab_guarded",
    "open_vocab_guarded_v2",
    "open_vocab_guarded_v3_relations",
    "open_vocab_guarded_v4_axioms",
    "open_vocab_guarded_v5_axioms_only",
    "open_vocab_guarded_v6_predicate_routed",
)
GUARDED_V2_SYSTEMS = {
    "open_vocab_guarded_v2",
    "open_vocab_guarded_v3_relations",
    "open_vocab_guarded_v4_axioms",
    "open_vocab_guarded_v5_axioms_only",
    "open_vocab_guarded_v6_predicate_routed",
}
GUARDED_SYSTEMS = {"open_vocab_guarded", *GUARDED_V2_SYSTEMS}
INSTANCE_RELATIONS = {"instance-of", "instanceof", "instance_of"}
TAXONOMY_RELATIONS = {"is-a", "isa", "subclass-of", "subclassof", "kind-of"}
GENERIC_RELATIONS = {
    "related to",
    "related-to",
    "associated with",
    "associated-with",
    "linked to",
    "linked-to",
    "for",
    "when",
    "captures",
    "aggregates",
    "determines",
    "defines",
    "bears",
    "facilitates",
    "expands",
    "enables",
    "accumulates",
    "supports",
    "optimized_for",
    "responsible_for",
}
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DEFAULT_IMPLICIT_CONFIDENCE_THRESHOLD = 0.95
DEFAULT_CALIBRATED_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_MAX_NONTAX = 8
DEFAULT_MAX_IMPLICIT_GAP_CHARS = 600
GUARDED_V2_POSTPROCESS_VERSION = "guarded_v2_postprocess_v3"

TASK_CONFIG = {
    "a": {
        "path": CHALLENGE / "TaskA-Flagship" / "train_task_a.json",
        "gold_key": "primitive-ontology-triples",
        "name": "Flagship",
    },
    "b": {
        "path": CHALLENGE / "TaskB-Reuse" / "train_task_b.json",
        "gold_key": "extended-primitive-ontology-triples",
        "name": "Reuse",
    },
}


def _norm(value: Any) -> str:
    return " ".join(str(value).lower().strip().split())


def normalize_allowed_nontax_predicates(
    value: str | Iterable[str] | None,
) -> frozenset[str] | None:
    """Return a canonical allowlist, or None when predicate filtering is disabled."""
    if value is None:
        return None
    raw_values = value.split(",") if isinstance(value, str) else value
    normalized: set[str] = set()
    for raw in raw_values:
        text = str(raw).strip()
        if not text:
            continue
        predicate, _, _ = canonicalize_predicate(text)
        normalized.add(predicate)
    return frozenset(normalized) if normalized else None


def calibrator_from_artifact_payload(
    payload: dict[str, Any],
) -> BetaIsotonicCalibrator:
    calibrator = BetaIsotonicCalibrator.from_dict(
        payload.get("calibrator", payload)
    )
    support = payload.get("deployment_support")
    if (
        calibrator.deployable
        and isinstance(support, dict)
        and support.get("deployable") is False
    ):
        calibrator.status = str(
            support.get("status") or "insufficient_independent_positive_samples"
        )
    return calibrator


def normalize_triples(items: Iterable[Any]) -> list[list[str]]:
    triples: list[list[str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items or []:
        if isinstance(item, dict):
            s = item.get("subject", item.get("head", item.get("child")))
            p = item.get("predicate", item.get("relation"))
            o = item.get("object", item.get("tail", item.get("parent")))
            values = (s, p, o)
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            values = item[:3]
        else:
            continue
        if any(v is None or not str(v).strip() for v in values):
            continue
        triple = tuple(_norm(v) for v in values)
        if triple not in seen:
            triples.append(list(triple))
            seen.add(triple)
    return triples


def parse_model_items(text: str) -> list[Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    payload: Any = None
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[{]", cleaned):
            try:
                payload, _ = decoder.raw_decode(cleaned[match.start() :])
                break
            except json.JSONDecodeError:
                continue
    if payload is None:
        raise ValueError("model response does not contain valid JSON")
    if isinstance(payload, dict):
        for key in (
            "triples",
            "primitive-ontology-triples",
            "extended-primitive-ontology-triples",
            "relations",
            "taxonomy",
        ):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            raise ValueError("model JSON object has no supported triple list")
    if not isinstance(payload, list):
        raise ValueError("model JSON must be a list or contain a triple list")
    return payload


def parse_model_payload(text: str) -> list[list[str]]:
    return normalize_triples(parse_model_items(text))


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(value))


def _evidence_supports_endpoints(evidence: str, subject: str, obj: str) -> bool:
    compact_evidence = _compact(evidence)
    return bool(
        compact_evidence
        and _compact(subject) in compact_evidence
        and _compact(obj) in compact_evidence
    )


def parse_guarded_model_payload(
    text: str,
    *,
    context: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_nontax: int = DEFAULT_MAX_NONTAX,
) -> tuple[list[list[str]], dict[str, Any]]:
    items = parse_model_items(text)
    context_norm = _norm(context)
    accepted_core: list[list[str]] = []
    accepted_nontax: list[dict[str, Any]] = []
    rejected_nontax: list[dict[str, Any]] = []
    raw_nontax = 0
    seen: set[tuple[str, str, str]] = set()

    for position, item in enumerate(items):
        if isinstance(item, dict):
            subject = item.get("subject", item.get("head", item.get("child")))
            predicate = item.get("predicate", item.get("relation"))
            obj = item.get("object", item.get("tail", item.get("parent")))
            confidence_raw = item.get("confidence")
            evidence = str(item.get("evidence") or "").strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            subject, predicate, obj = item[:3]
            confidence_raw = None
            evidence = ""
        else:
            continue
        if any(value is None or not str(value).strip() for value in (subject, predicate, obj)):
            continue

        triple = (_norm(subject), _norm(predicate), _norm(obj))
        if triple in seen:
            continue
        seen.add(triple)
        if triple[1] in INSTANCE_RELATIONS or triple[1] in TAXONOMY_RELATIONS:
            accepted_core.append(list(triple))
            continue
        raw_nontax += 1

        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = -1.0
        reasons: list[str] = []
        evidence_norm = _norm(evidence)
        if not 0.0 <= confidence <= 1.0:
            reasons.append("invalid_confidence")
        elif confidence < confidence_threshold:
            reasons.append("low_confidence")
        if not evidence_norm:
            reasons.append("missing_evidence")
        elif evidence_norm not in context_norm:
            reasons.append("evidence_not_verbatim")
        elif not _evidence_supports_endpoints(evidence, triple[0], triple[2]):
            reasons.append("evidence_missing_endpoints")
        if triple[0] == triple[2]:
            reasons.append("self_loop")
        if triple[1] in GENERIC_RELATIONS:
            reasons.append("generic_predicate")

        candidate = {
            "triple": list(triple),
            "confidence": confidence if confidence >= 0.0 else None,
            "evidence": evidence,
            "position": position,
        }
        if reasons:
            candidate["reasons"] = reasons
            rejected_nontax.append(candidate)
        else:
            accepted_nontax.append(candidate)

    accepted_nontax.sort(key=lambda item: (-float(item["confidence"]), int(item["position"])))
    kept_nontax = accepted_nontax[: max(0, max_nontax)]
    for candidate in accepted_nontax[max(0, max_nontax) :]:
        rejected = dict(candidate)
        rejected["reasons"] = ["quantity_limit"]
        rejected_nontax.append(rejected)

    triples = accepted_core + [candidate["triple"] for candidate in kept_nontax]
    reason_counts: dict[str, int] = {}
    for candidate in rejected_nontax:
        for reason in candidate["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    diagnostics = {
        "confidence_threshold": confidence_threshold,
        "max_nontax": max_nontax,
        "raw_candidates": len(items),
        "accepted_core": len(accepted_core),
        "raw_nontax": raw_nontax,
        "accepted_nontax": len(kept_nontax),
        "rejected_nontax": len(rejected_nontax),
        "rejection_reason_counts": reason_counts,
        "accepted_nontax_candidates": kept_nontax,
        "rejected_nontax_candidates": rejected_nontax,
    }
    return triples, diagnostics


def _evidence_spans(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _implicit_evidence_gap(context: str, spans: list[str]) -> int | None:
    context_norm = _norm(context)
    positions: list[tuple[int, int]] = []
    for span in spans:
        span_norm = _norm(span)
        start = context_norm.find(span_norm)
        if start < 0:
            return None
        positions.append((start, start + len(span_norm)))
    positions.sort()
    return max(0, positions[-1][0] - positions[0][1])


def parse_guarded_v2_model_payload(
    text: str,
    *,
    context: str,
    sample: dict[str, Any],
    task: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    implicit_confidence_threshold: float = DEFAULT_IMPLICIT_CONFIDENCE_THRESHOLD,
    calibrated_confidence_threshold: float = DEFAULT_CALIBRATED_CONFIDENCE_THRESHOLD,
    max_nontax: int = DEFAULT_MAX_NONTAX,
    max_implicit_gap_chars: int = DEFAULT_MAX_IMPLICIT_GAP_CHARS,
    calibrator: BetaIsotonicCalibrator | None = None,
    allowed_nontax_predicates: str | Iterable[str] | None = None,
) -> tuple[list[list[str]], dict[str, Any]]:
    """Parse guarded-v2 output with canonicalization and two-span implicit evidence."""
    items = parse_model_items(text)
    context_norm = _norm(context)
    entity_inventory = build_entity_inventory(sample, task=task)
    accepted_core: list[list[str]] = []
    accepted_nontax: list[dict[str, Any]] = []
    rejected_nontax: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    raw_nontax = 0
    canonicalized_candidates = 0
    calibration_status = calibrator.status if calibrator is not None else "not_configured"
    use_calibration = calibrator is not None and calibrator.deployable
    allowed_predicates = normalize_allowed_nontax_predicates(allowed_nontax_predicates)

    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        subject = item.get("subject", item.get("head", item.get("child")))
        predicate = item.get("predicate", item.get("relation"))
        obj = item.get("object", item.get("tail", item.get("parent")))
        if any(value is None or not str(value).strip() for value in (subject, predicate, obj)):
            continue

        canonical, canonicalization = canonicalize_triple(
            (subject, predicate, obj),
            context=context,
            entity_inventory=entity_inventory,
        )
        triple = tuple(canonical)
        if triple in seen:
            continue
        seen.add(triple)
        if canonicalization["changed"]:
            canonicalized_candidates += 1
        if triple[1] in INSTANCE_RELATIONS or triple[1] in TAXONOMY_RELATIONS:
            accepted_core.append(list(triple))
            continue
        raw_nontax += 1

        try:
            raw_confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            raw_confidence = -1.0
        evidence = _evidence_spans(item.get("evidence"))
        evidence_mode = _norm(item.get("evidence_mode") or "")
        if not evidence_mode:
            evidence_mode = "explicit" if len(evidence) == 1 else "cross_sentence"
        calibrated_confidence = (
            calibrator.calibrate(raw_confidence) if use_calibration and raw_confidence >= 0.0 else None
        )
        ranking_confidence = (
            float(calibrated_confidence) if calibrated_confidence is not None else raw_confidence
        )
        reasons: list[str] = []
        evidence_gap_chars: int | None = None

        if not 0.0 <= raw_confidence <= 1.0:
            reasons.append("invalid_confidence")
        else:
            raw_threshold = (
                implicit_confidence_threshold
                if evidence_mode == "cross_sentence"
                else confidence_threshold
            )
            if raw_confidence < raw_threshold:
                reasons.append("low_confidence")
            if (
                calibrated_confidence is not None
                and calibrated_confidence < calibrated_confidence_threshold
            ):
                reasons.append("low_calibrated_confidence")

        if evidence_mode == "explicit":
            if len(evidence) != 1:
                reasons.append("explicit_evidence_count")
            elif _norm(evidence[0]) not in context_norm:
                reasons.append("evidence_not_verbatim")
            elif not _evidence_supports_endpoints(evidence[0], triple[0], triple[2]):
                reasons.append("evidence_missing_endpoints")
        elif evidence_mode == "cross_sentence":
            if len(evidence) != 2:
                reasons.append("implicit_evidence_count")
            elif _norm(evidence[0]) == _norm(evidence[1]):
                reasons.append("implicit_evidence_duplicate")
            elif any(_norm(span) not in context_norm for span in evidence):
                reasons.append("evidence_not_verbatim")
            else:
                combined = " ".join(evidence)
                if not _evidence_supports_endpoints(combined, triple[0], triple[2]):
                    reasons.append("evidence_missing_endpoints")
                if any(
                    not (
                        _compact(triple[0]) in _compact(span)
                        or _compact(triple[2]) in _compact(span)
                    )
                    for span in evidence
                ):
                    reasons.append("implicit_span_missing_endpoint")
                gap = _implicit_evidence_gap(context, evidence)
                evidence_gap_chars = gap
                if gap is None:
                    reasons.append("evidence_not_verbatim")
                elif gap > max_implicit_gap_chars:
                    reasons.append("implicit_evidence_too_distant")
        else:
            reasons.append("invalid_evidence_mode")

        if triple[0] == triple[2]:
            reasons.append("self_loop")
        if triple[1] in GENERIC_RELATIONS:
            reasons.append("generic_predicate")
        if allowed_predicates is not None and triple[1] not in allowed_predicates:
            reasons.append("predicate_not_calibration_supported")

        candidate = {
            "triple": list(triple),
            "raw_confidence": raw_confidence if raw_confidence >= 0.0 else None,
            "calibrated_confidence": calibrated_confidence,
            "ranking_confidence": ranking_confidence if ranking_confidence >= 0.0 else None,
            "evidence": evidence,
            "evidence_mode": evidence_mode,
            "evidence_gap_chars": evidence_gap_chars,
            "canonicalization": canonicalization,
            "position": position,
        }
        if reasons:
            candidate["reasons"] = list(dict.fromkeys(reasons))
            rejected_nontax.append(candidate)
        else:
            accepted_nontax.append(candidate)

    accepted_nontax.sort(
        key=lambda item: (-float(item["ranking_confidence"]), int(item["position"]))
    )
    kept_nontax = accepted_nontax[: max(0, max_nontax)]
    for candidate in accepted_nontax[max(0, max_nontax) :]:
        rejected = dict(candidate)
        rejected["reasons"] = ["quantity_limit"]
        rejected_nontax.append(rejected)

    reason_counts: dict[str, int] = {}
    for candidate in rejected_nontax:
        for reason in candidate["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    diagnostics = {
        "canonicalization_version": CANONICALIZATION_VERSION,
        "postprocess_version": GUARDED_V2_POSTPROCESS_VERSION,
        "calibration_status": calibration_status,
        "calibration_applied": use_calibration,
        "confidence_threshold": confidence_threshold,
        "implicit_confidence_threshold": implicit_confidence_threshold,
        "calibrated_confidence_threshold": calibrated_confidence_threshold,
        "allowed_nontax_predicates": (
            sorted(allowed_predicates) if allowed_predicates is not None else None
        ),
        "max_implicit_gap_chars": max_implicit_gap_chars,
        "max_nontax": max_nontax,
        "raw_candidates": len(items),
        "canonicalized_candidates": canonicalized_candidates,
        "accepted_core": len(accepted_core),
        "raw_nontax": raw_nontax,
        "accepted_nontax": len(kept_nontax),
        "rejected_nontax": len(rejected_nontax),
        "rejection_reason_counts": reason_counts,
        "accepted_nontax_candidates": kept_nontax,
        "rejected_nontax_candidates": rejected_nontax,
    }
    triples = accepted_core + [candidate["triple"] for candidate in kept_nontax]
    return triples, diagnostics


def build_prompt(
    sample: dict[str, Any],
    task: str,
    system: str,
    max_context_chars: int,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_nontax: int = DEFAULT_MAX_NONTAX,
    implicit_confidence_threshold: float = DEFAULT_IMPLICIT_CONFIDENCE_THRESHOLD,
) -> str:
    if task not in TASK_CONFIG:
        raise ValueError(f"unsupported task: {task}")
    if system not in SYSTEMS:
        raise ValueError(f"unsupported system: {system}")
    context = str(sample.get("context") or "")[:max_context_chars]
    if system in GUARDED_V2_SYSTEMS:
        relation_only = system in {
            "open_vocab_guarded_v3_relations",
            "open_vocab_guarded_v4_axioms",
            "open_vocab_guarded_v5_axioms_only",
            "open_vocab_guarded_v6_predicate_routed",
        }
        route = route_for_system(system)
        route_labels = inventory_labels(route.key)
        preferred_predicates = render_labels(route_labels, style=route.style)
        common = (
            "You are an ontology learning system. Return compact JSON only, with this exact shape: "
            '{"triples":[{"subject":"...","predicate":"...","object":"...",'
            '"confidence":0.0,"evidence_mode":"explicit|cross_sentence",'
            '"evidence":["verbatim quote 1","optional verbatim quote 2"]}]}. '
            "Use lowercase subject, predicate, and object strings; avoid duplicate triples; "
            "and emit at most 80 triples in total. Prefer canonical ontology predicates such as "
            f"{preferred_predicates} when their "
            "meaning and direction match the context. For an explicit non-taxonomic relation, use "
            "evidence_mode explicit and exactly one verbatim span containing both endpoints. For an "
            "implicit relation supported across nearby sentences, use evidence_mode cross_sentence "
            "and exactly two verbatim spans; each span must contain at least one endpoint and their "
            "combination must entail the relation. Do not use background knowledge or merely plausible "
            "inferences. Use stable reusable predicates, not narrative verbs or prepositions. "
            f"Emit at most {max_nontax} non-taxonomic relations. Explicit candidates require raw "
            f"confidence at least {confidence_threshold:.2f}; cross-sentence candidates require at "
            f"least {implicit_confidence_threshold:.2f}. Confidence must be a number from 0 to 1.\n"
        )
        if relation_only:
            if system == "open_vocab_guarded_v6_predicate_routed":
                common += (
                    "RELATION-ONLY FROZEN PREDICATE-FAMILY ROUTER: emit only the exact "
                    "predicate strings in this inventory: "
                    f"{render_labels(route_labels, style=STYLE_SEMICOLON)}. First identify which of "
                    "these families the text explicitly supports, then emit only supported "
                    "triples. AXIOM FAMILY: use equivalent class only for interchangeable "
                    "class names and disjoint with only for explicitly incompatible classes. "
                    "TYPE/METADATA FAMILY: use type for an entity-to-category assertion, "
                    "database_cross_reference for a stated database or canonical-name cross "
                    "reference, has role for an explicit role, and same as only for explicit "
                    "identity. CHEMISTRY FAMILY: is_conjugate_acid_of points from an acid to "
                    "its conjugate base; is_conjugate_base_of points from a base to its "
                    "conjugate acid. Never emit taxonomy, instance-of, part-whole, location, "
                    "derivation, development, regulation, narrative verbs, or any predicate "
                    "outside the frozen inventory. Do not infer disjointness from ordinary "
                    "difference or equivalence from subtype. Return {\"triples\":[]} when no "
                    "inventory relation is explicitly supported.\n"
                )
            elif system == "open_vocab_guarded_v5_axioms_only":
                common += (
                    "RELATION-ONLY AXIOM-INVENTORY PASS: emit only "
                    f"{render_labels(route_labels, style=STYLE_OR)} "
                    "triples; never emit any other predicate, including part-whole, location, "
                    "derivation, taxonomy, or typing. Treat two classes as equivalent only when "
                    "the context presents them as interchangeable names for the same category. "
                    "Treat classes as disjoint only when the context presents mutually exclusive "
                    "contrasts or an exhaustive division into incompatible categories. Do not "
                    "infer equivalence from an ordinary subtype statement, or disjointness merely "
                    "because two entities are different siblings. Return {\"triples\":[]} when "
                    "no supported axiom exists.\n"
                )
            elif system == "open_vocab_guarded_v4_axioms":
                common += (
                    "RELATION-ONLY AXIOM PASS: emit non-taxonomic relations only; never emit "
                    "instance-of, is-a, subclass, typing, or taxonomy triples. Systematically "
                    "scan for equivalence, disjointness, part-whole, origin/derivation, "
                    "developmental-origin, location, and regulation assertions. Use canonical "
                    "predicates equivalent class and disjoint with for ontology axioms. Do not "
                    "infer equivalence from an ordinary subtype statement, or disjointness merely "
                    "because two entities are different siblings. Return {\"triples\":[]} when "
                    "no supported relation exists.\n"
                )
            else:
                common += (
                    "RELATION-ONLY PASS: emit non-taxonomic relations only; never emit instance-of, "
                    "is-a, subclass, typing, or taxonomy triples. Systematically scan for explicit "
                    "part-whole, origin/derivation, developmental-origin, location, and regulation "
                    "assertions before returning an empty list. Use part_of versus has part and "
                    "develops_from versus derives from only when their direction and meaning exactly "
                    "match the context. Return {\"triples\":[]} when no supported relation exists.\n"
                )
    elif system == "open_vocab_guarded":
        common = (
            "You are an ontology learning system. Return compact JSON only, with this exact shape: "
            '{"triples":[{"subject":"...","predicate":"...","object":"...",'
            '"confidence":0.0,"evidence":"verbatim quote from CONTEXT"}]}. '
            "Use lowercase subject, predicate, and object strings; avoid duplicate triples; "
            "and emit at most 80 triples in total. Output instance-of and is-a triples, plus "
            "non-taxonomic relations only when a single verbatim context span directly supports "
            "the subject-predicate-object assertion. The evidence span must contain both the "
            "subject and object. A non-taxonomic predicate must name a stable, reusable ontology "
            "relation; omit narrative verbs, prepositions, inferred, generic, or merely plausible "
            "relations. Do not turn every verb phrase in the context into a relation. "
            f"Emit at most {max_nontax} non-taxonomic relations and omit non-taxonomic candidates "
            f"below confidence {confidence_threshold:.2f}. Confidence must be a number from 0 to 1.\n"
        )
    else:
        relation_policy = (
            "Output only instance-of and is-a triples. Do not output any other predicate."
            if system == "closed_vocab"
            else (
                "Output instance-of and is-a triples, plus non-taxonomic relations that are "
                "explicitly supported by the context. Use concise predicate phrases from the text."
            )
        )
        common = (
            "You are an ontology learning system. Return compact JSON only, with this exact shape: "
            '{"triples":[["subject","predicate","object"]]}. '
            "Use lowercase strings, avoid duplicate triples, and emit at most 80 triples. "
            f"{relation_policy}\n"
        )
    if task == "a":
        return (
            common
            + "Task A Flagship: construct a primitive ontology using only the supplied context. "
            "Do not assume a hidden term or type inventory.\n\n"
            + f"CONTEXT:\n{context}"
        )

    initial = normalize_triples(sample.get("initial-primitive-ontology-triples") or [])
    given_terms = [_norm(x) for x in (sample.get("terms") or []) if str(x).strip()]
    given_types = [_norm(x) for x in (sample.get("types") or []) if str(x).strip()]
    return (
        common
        + "Task B Reuse: predict extension triples only. Do not repeat triples already present "
        "in the initial ontology. The terms/types below are official task inputs, not gold targets.\n\n"
        + f"CONTEXT:\n{context}\n\n"
        + f"INITIAL ONTOLOGY:\n{json.dumps(initial, ensure_ascii=False)}\n\n"
        + f"GIVEN TERMS:\n{json.dumps(given_terms, ensure_ascii=False)}\n\n"
        + f"GIVEN TYPES:\n{json.dumps(given_types, ensure_ascii=False)}"
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_") or "model"


def postprocess_sha256(
    *,
    confidence_threshold: float,
    implicit_confidence_threshold: float,
    calibrated_confidence_threshold: float,
    max_nontax: int,
    max_implicit_gap_chars: int,
    calibrator: BetaIsotonicCalibrator | None,
    allowed_nontax_predicates: str | Iterable[str] | None = None,
) -> str:
    allowed_predicates = normalize_allowed_nontax_predicates(allowed_nontax_predicates)
    return _sha256(
        json.dumps(
            {
                "postprocess_version": GUARDED_V2_POSTPROCESS_VERSION,
                "canonicalization_version": CANONICALIZATION_VERSION,
                "calibrator": calibrator.to_dict() if calibrator is not None else None,
                "confidence_threshold": confidence_threshold,
                "implicit_confidence_threshold": implicit_confidence_threshold,
                "calibrated_confidence_threshold": calibrated_confidence_threshold,
                "allowed_nontax_predicates": (
                    sorted(allowed_predicates) if allowed_predicates is not None else None
                ),
                "max_nontax": max_nontax,
                "max_implicit_gap_chars": max_implicit_gap_chars,
            },
            sort_keys=True,
        )
    )


def cache_is_replayable(
    path: Path,
    *,
    system: str,
    prompt_sha256: str,
    postprocess_hash: str | None = None,
) -> bool:
    if not cache_has_reusable_generation(path, prompt_sha256=prompt_sha256):
        return False
    if system not in GUARDED_V2_SYSTEMS:
        return True
    cached = json.loads(path.read_text(encoding="utf-8"))
    return cached.get("postprocess_sha256") == postprocess_hash


def cache_has_reusable_generation(path: Path, *, prompt_sha256: str) -> bool:
    if not path.exists():
        return False
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        cached.get("protocol") == PROTOCOL
        and cached.get("prompt_sha256") == prompt_sha256
        and cached.get("status") == "ok"
        and isinstance(cached.get("raw_text"), str)
    )


def previous_attempt_state(
    path: Path, *, prompt_sha256: str
) -> tuple[int, list[str]]:
    """Return retry provenance only for the same frozen prompt identity."""
    if not path.exists():
        return 0, []
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, []
    if (
        cached.get("protocol") != PROTOCOL
        or cached.get("prompt_sha256") != prompt_sha256
    ):
        return 0, []
    errors = list(cached.get("errors") or cached.get("previous_errors") or [])
    return int(cached.get("attempts") or 0), errors


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def cache_path(model: str, task: str, system: str, index: int, sample_id: str) -> Path:
    id_hash = _sha256(sample_id)[:10]
    return (
        RESULTS
        / "blind_cache"
        / _slug(model)
        / f"task_{task}"
        / system
        / f"{index:05d}_{id_hash}.json"
    )


def categorize(triples: Iterable[Iterable[str]]) -> dict[str, set[tuple]]:
    typing: set[tuple[str, str]] = set()
    taxonomy: set[tuple[str, str]] = set()
    nontax: set[tuple[str, str, str]] = set()
    for raw in triples:
        values = list(raw)
        if len(values) < 3:
            continue
        s, p, o = (_norm(values[0]), _norm(values[1]), _norm(values[2]))
        if p in INSTANCE_RELATIONS:
            typing.add((s, o))
        elif p in TAXONOMY_RELATIONS:
            taxonomy.add((s, o))
        else:
            nontax.add((s, p, o))
    return {"typing": typing, "taxonomy": taxonomy, "nontax": nontax}


def prf(gold: set[tuple], pred: set[tuple]) -> dict[str, float | int]:
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_sample(gold: list[list[str]], pred: list[list[str]]) -> dict[str, Any]:
    gold_c = categorize(gold)
    pred_c = categorize(pred)
    gold_nontax_nonself = {triple for triple in gold_c["nontax"] if triple[0] != triple[2]}
    pred_nontax_nonself = {triple for triple in pred_c["nontax"] if triple[0] != triple[2]}
    graph = exact_match(gold, pred)
    return {
        "typing": prf(gold_c["typing"], pred_c["typing"]),
        "taxonomy": prf(gold_c["taxonomy"], pred_c["taxonomy"]),
        "nontax": prf(gold_c["nontax"], pred_c["nontax"]),
        "nontax_nonself": prf(gold_nontax_nonself, pred_nontax_nonself),
        "official_graph_similarity": graph,
        "gold_counts": {k: len(v) for k, v in gold_c.items()},
        "pred_counts": {k: len(v) for k, v in pred_c.items()},
    }


def aggregate_records(records: list[dict[str, Any]], target_n: int) -> dict[str, Any]:
    successful = [r for r in records if r.get("status") == "ok"]
    latencies = sorted(
        float(r["latency_sec"])
        for r in successful
        if r.get("latency_sec") is not None
    )

    def percentile_nearest_rank(fraction: float) -> float:
        if not latencies:
            return 0.0
        rank = max(1, math.ceil(fraction * len(latencies)))
        return latencies[min(rank - 1, len(latencies) - 1)]

    def merge_metric(key: str) -> dict[str, float | int]:
        tp = sum(int(r["metrics"][key]["tp"]) for r in successful)
        fp = sum(int(r["metrics"][key]["fp"]) for r in successful)
        fn = sum(int(r["metrics"][key]["fn"]) for r in successful)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    typing = merge_metric("typing")
    taxonomy = merge_metric("taxonomy")
    nontax = merge_metric("nontax")
    nontax_nonself = merge_metric("nontax_nonself")
    graph = (
        sum(float(r["metrics"]["official_graph_similarity"]["graph_similarity"]) for r in successful)
        / len(successful)
        if successful
        else 0.0
    )
    summary = {
        "target_n": target_n,
        "n_success": len(successful),
        "n_failed": len(records) - len(successful),
        "complete": len(successful) == target_n,
        "term_typing_f1_micro": typing,
        "taxonomy_f1_micro": taxonomy,
        "nontax_f1_micro": nontax,
        "nontax_nonself_f1_micro": nontax_nonself,
        "official_graph_similarity_avg": graph,
        "composite_typing_tax_gs": (float(typing["f1"]) + float(taxonomy["f1"]) + graph) / 3.0,
        "nontax_pred_total": sum(r["metrics"]["pred_counts"]["nontax"] for r in successful),
        "nontax_gold_total": sum(r["metrics"]["gold_counts"]["nontax"] for r in successful),
        "nontax_nonself_pred_total": int(nontax_nonself["tp"]) + int(nontax_nonself["fp"]),
        "nontax_nonself_gold_total": int(nontax_nonself["tp"]) + int(nontax_nonself["fn"]),
        "successful_latency_sec": {
            "min": round(latencies[0], 3) if latencies else 0.0,
            "median": round(statistics.median(latencies), 3) if latencies else 0.0,
            "p95": round(percentile_nearest_rank(0.95), 3),
            "max": round(latencies[-1], 3) if latencies else 0.0,
            "sum": round(sum(latencies), 3),
        },
        "successful_runner_attempts_total": sum(
            int(r.get("attempts") or 0) for r in successful
        ),
        "successful_samples_with_retry": sum(
            1 for r in successful if int(r.get("attempts") or 0) > 1
        ),
    }
    guarded = [r["guard_diagnostics"] for r in successful if r.get("guard_diagnostics")]
    if guarded:
        reason_counts: dict[str, int] = {}
        for diagnostics in guarded:
            for reason, count in diagnostics.get("rejection_reason_counts", {}).items():
                reason_counts[reason] = reason_counts.get(reason, 0) + int(count)
        summary["guard"] = {
            "samples": len(guarded),
            "raw_nontax": sum(int(d.get("raw_nontax", 0)) for d in guarded),
            "accepted_nontax": sum(int(d.get("accepted_nontax", 0)) for d in guarded),
            "rejected_nontax": sum(int(d.get("rejected_nontax", 0)) for d in guarded),
            "rejection_reason_counts": reason_counts,
        }
    return summary


def predict_one(
    *,
    llm: Any,
    sample: dict[str, Any],
    index: int,
    task: str,
    system: str,
    model: str,
    max_context_chars: int,
    confidence_threshold: float,
    implicit_confidence_threshold: float,
    calibrated_confidence_threshold: float,
    max_nontax: int,
    max_implicit_gap_chars: int,
    calibrator: BetaIsotonicCalibrator | None,
    parse_retries: int,
    refresh_cache: bool,
    allowed_nontax_predicates: str | Iterable[str] | None = None,
) -> dict[str, Any]:
    allowed_predicates = normalize_allowed_nontax_predicates(
        allowed_nontax_predicates
    )
    prompt = build_prompt(
        sample,
        task,
        system,
        max_context_chars,
        confidence_threshold,
        max_nontax,
        implicit_confidence_threshold,
    )
    prompt_hash = _sha256(prompt)
    postprocess_hash = postprocess_sha256(
        confidence_threshold=confidence_threshold,
        implicit_confidence_threshold=implicit_confidence_threshold,
        calibrated_confidence_threshold=calibrated_confidence_threshold,
        max_nontax=max_nontax,
        max_implicit_gap_chars=max_implicit_gap_chars,
        calibrator=calibrator,
        allowed_nontax_predicates=allowed_predicates,
    )
    sample_id = str(sample.get("id") or index)
    path = cache_path(model, task, system, index, sample_id)
    if not refresh_cache and cache_has_reusable_generation(path, prompt_sha256=prompt_hash):
        cached = json.loads(path.read_text(encoding="utf-8"))
        if system in GUARDED_V2_SYSTEMS and cached.get("postprocess_sha256") != postprocess_hash:
            triples, guard_diagnostics = parse_guarded_v2_model_payload(
                cached["raw_text"],
                context=str(sample.get("context") or "")[:max_context_chars],
                sample=sample,
                task=task,
                confidence_threshold=confidence_threshold,
                implicit_confidence_threshold=implicit_confidence_threshold,
                calibrated_confidence_threshold=calibrated_confidence_threshold,
                max_nontax=max_nontax,
                max_implicit_gap_chars=max_implicit_gap_chars,
                calibrator=calibrator,
                allowed_nontax_predicates=allowed_predicates,
            )
            cached["triples"] = triples
            cached["guard_diagnostics"] = guard_diagnostics
            cached["postprocess_sha256"] = postprocess_hash
            cached.setdefault("generation_config", {}).update(
                {
                    "confidence_threshold": confidence_threshold,
                    "implicit_confidence_threshold": implicit_confidence_threshold,
                    "calibrated_confidence_threshold": calibrated_confidence_threshold,
                    "allowed_nontax_predicates": (
                        sorted(allowed_predicates)
                        if allowed_predicates is not None
                        else None
                    ),
                    "max_nontax": max_nontax,
                    "max_implicit_gap_chars": max_implicit_gap_chars,
                    "calibration_status": (
                        calibrator.status if calibrator is not None else "not_configured"
                    ),
                }
            )
            cached["locally_reprocessed_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(path, cached)
        return cached

    previous_attempts, previous_errors = previous_attempt_state(
        path, prompt_sha256=prompt_hash
    )

    started = time.perf_counter()
    errors: list[str] = []
    raw_text = ""
    for attempt in range(1, max(1, parse_retries) + 1):
        try:
            suffix = ""
            if attempt > 1:
                suffix = "\n\nYour previous response was invalid. Return the required JSON object only."
            response = llm.complete(
                prompt + suffix,
                task=f"official_blind_{task}_{system}",
                meta={"sample_id": sample_id, "sample_index": index},
            )
            raw_text = response.text
            guard_diagnostics = None
            if system in GUARDED_V2_SYSTEMS:
                triples, guard_diagnostics = parse_guarded_v2_model_payload(
                    raw_text,
                    context=str(sample.get("context") or "")[:max_context_chars],
                    sample=sample,
                    task=task,
                    confidence_threshold=confidence_threshold,
                    implicit_confidence_threshold=implicit_confidence_threshold,
                    calibrated_confidence_threshold=calibrated_confidence_threshold,
                    max_nontax=max_nontax,
                    max_implicit_gap_chars=max_implicit_gap_chars,
                    calibrator=calibrator,
                    allowed_nontax_predicates=allowed_predicates,
                )
            elif system == "open_vocab_guarded":
                triples, guard_diagnostics = parse_guarded_model_payload(
                    raw_text,
                    context=str(sample.get("context") or "")[:max_context_chars],
                    confidence_threshold=confidence_threshold,
                    max_nontax=max_nontax,
                )
            else:
                triples = parse_model_payload(raw_text)
            record = {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "status": "ok",
                "task": task,
                "system": system,
                "model": model,
                "sample_index": index,
                "sample_id": sample_id,
                "prompt_sha256": prompt_hash,
                "postprocess_sha256": postprocess_hash,
                "generation_config": {
                    "max_completion_tokens": int(
                        os.environ.get("ONTOBRIDG_MAX_TOKENS") or 1024
                    ),
                    "confidence_threshold": confidence_threshold,
                    "implicit_confidence_threshold": implicit_confidence_threshold,
                    "calibrated_confidence_threshold": calibrated_confidence_threshold,
                    "allowed_nontax_predicates": (
                        sorted(allowed_predicates)
                        if allowed_predicates is not None
                        else None
                    ),
                    "max_nontax": max_nontax,
                    "max_implicit_gap_chars": max_implicit_gap_chars,
                    "calibration_status": (
                        calibrator.status if calibrator is not None else "not_configured"
                    ),
                },
                "triples": triples,
                "raw_text": raw_text,
                "attempts": previous_attempts + attempt,
                "attempts_this_run": attempt,
                "previous_attempts": previous_attempts,
                "previous_errors": previous_errors,
                "attempt_errors_this_run": list(errors),
                "errors": previous_errors + errors,
                "latency_sec": round(time.perf_counter() - started, 3),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            if guard_diagnostics is not None:
                record["guard_diagnostics"] = guard_diagnostics
            atomic_write_json(path, record)
            return record
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")

    record = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "error",
        "task": task,
        "system": system,
        "model": model,
        "sample_index": index,
        "sample_id": sample_id,
        "prompt_sha256": prompt_hash,
        "raw_text": raw_text,
        "errors": previous_errors + errors,
        "attempts": previous_attempts + max(1, parse_retries),
        "attempts_this_run": max(1, parse_retries),
        "previous_attempts": previous_attempts,
        "previous_errors": previous_errors,
        "attempt_errors_this_run": errors,
        "latency_sec": round(time.perf_counter() - started, 3),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(path, record)
    return record


def load_samples(task: str, offset: int, limit: int) -> list[tuple[int, dict[str, Any]]]:
    config = TASK_CONFIG[task]
    data = json.loads(config["path"].read_text(encoding="utf-8"))
    stop = len(data) if limit <= 0 else min(len(data), offset + limit)
    return list(enumerate(data[offset:stop], start=offset))


def run_system(
    *,
    args: argparse.Namespace,
    task: str,
    system: str,
    indexed_samples: list[tuple[int, dict[str, Any]]],
    checkpoint_path: Path,
) -> dict[str, Any]:
    llm = build_llm(args.backend, model=args.model, base_url=args.base_url)
    calibrator: BetaIsotonicCalibrator | None = None
    if system in GUARDED_V2_SYSTEMS and args.calibration_file:
        calibration_path = Path(args.calibration_file)
        calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibrator = calibrator_from_artifact_payload(calibration_payload)
    allowed_nontax_predicates = normalize_allowed_nontax_predicates(
        getattr(args, "allowed_nontax_predicates", None)
    )
    records_by_index: dict[int, dict[str, Any]] = {}
    started = time.perf_counter()

    def run_item(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, sample = item
        prediction = predict_one(
            llm=llm,
            sample=sample,
            index=index,
            task=task,
            system=system,
            model=args.model,
            max_context_chars=args.max_context_chars,
            confidence_threshold=args.confidence_threshold,
            implicit_confidence_threshold=args.implicit_confidence_threshold,
            calibrated_confidence_threshold=args.calibrated_confidence_threshold,
            max_nontax=args.max_nontax,
            max_implicit_gap_chars=args.max_implicit_gap_chars,
            calibrator=calibrator,
            parse_retries=args.parse_retries,
            refresh_cache=args.refresh_cache,
            allowed_nontax_predicates=allowed_nontax_predicates,
        )
        record = {
            "sample_index": index,
            "sample_id": str(sample.get("id") or index),
            "status": prediction["status"],
            "cache": str(cache_path(args.model, task, system, index, str(sample.get("id") or index))),
            "attempts": prediction.get("attempts", 0),
            "latency_sec": prediction.get("latency_sec", 0.0),
        }
        if prediction["status"] == "ok":
            gold = normalize_triples(sample.get(TASK_CONFIG[task]["gold_key"]) or [])
            record["metrics"] = evaluate_sample(gold, prediction["triples"])
            if prediction.get("guard_diagnostics") is not None:
                record["guard_diagnostics"] = prediction["guard_diagnostics"]
        else:
            record["errors"] = prediction.get("errors", [])
        return record

    def snapshot() -> dict[str, Any]:
        records = [records_by_index[i] for i in sorted(records_by_index)]
        summary = aggregate_records(records, len(indexed_samples))
        payload = {
            "meta": {
                "challenge": "LLMs4OL 2026",
                "protocol": PROTOCOL,
                "input_policy": (
                    "context_only" if task == "a" else "context_initial_official_terms_types"
                ),
                "gold_visible_to_model": False,
                "task": task,
                "task_name": TASK_CONFIG[task]["name"],
                "system": system,
                "backend": args.backend,
                "model": args.model,
                "offset": args.offset,
                "limit": args.limit,
                "workers": args.workers,
                "max_context_chars": args.max_context_chars,
                "confidence_threshold": args.confidence_threshold,
                "implicit_confidence_threshold": args.implicit_confidence_threshold,
                "calibrated_confidence_threshold": args.calibrated_confidence_threshold,
                "allowed_nontax_predicates": (
                    sorted(allowed_nontax_predicates)
                    if allowed_nontax_predicates is not None
                    else None
                ),
                "max_nontax": args.max_nontax,
                "max_implicit_gap_chars": args.max_implicit_gap_chars,
                "calibration_file": args.calibration_file,
                "calibration_status": (
                    calibrator.status if calibrator is not None else "not_configured"
                ),
                "max_completion_tokens": int(os.environ.get("ONTOBRIDG_MAX_TOKENS") or 1024),
                "metric": "llms4ol2026_graph_similarity_exact",
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "elapsed_sec_scope": "current_process_including_cache_replay",
            },
            "summary": summary,
            "samples": records,
        }
        atomic_write_json(checkpoint_path, payload)
        return payload

    if args.workers == 1:
        for item in indexed_samples:
            record = run_item(item)
            records_by_index[record["sample_index"]] = record
            snapshot()
            if args.verbose:
                print(
                    f"  {system} sample={record['sample_index']} status={record['status']} "
                    f"latency={record['latency_sec']:.1f}s",
                    flush=True,
                )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_item, item): item[0] for item in indexed_samples}
            for future in as_completed(futures):
                record = future.result()
                records_by_index[record["sample_index"]] = record
                snapshot()
                if args.verbose:
                    print(
                        f"  {system} sample={record['sample_index']} status={record['status']} "
                        f"latency={record['latency_sec']:.1f}s",
                        flush=True,
                    )
    return snapshot()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["a", "b", "ab"], default="ab")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--systems", default="closed_vocab,open_vocab")
    parser.add_argument("--backend", choices=["openai", "openai_compat", "api"], default="openai")
    parser.add_argument("--model", default=os.environ.get("ONTOBRIDG_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument(
        "--implicit-confidence-threshold",
        type=float,
        default=DEFAULT_IMPLICIT_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument(
        "--calibrated-confidence-threshold",
        type=float,
        default=DEFAULT_CALIBRATED_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument("--max-nontax", type=int, default=DEFAULT_MAX_NONTAX)
    parser.add_argument(
        "--max-implicit-gap-chars", type=int, default=DEFAULT_MAX_IMPLICIT_GAP_CHARS
    )
    parser.add_argument("--calibration-file", default=None)
    parser.add_argument(
        "--allowed-nontax-predicates",
        default="",
        help="Optional comma-separated canonical predicate allowlist for precision mode.",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Write a content-free manifest of external payload scope without calling the API",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def build_audit_manifest(args: argparse.Namespace, tasks: list[str], systems: list[str]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "meta": {
            "protocol": PROTOCOL,
            "audit_only": True,
            "external_destination": (
                args.base_url
                or os.environ.get("ONTOBRIDG_BASE_URL")
                or "https://api.openai.com/v1"
            ),
            "model": args.model,
            "systems": systems,
            "gold_visible_to_model": False,
            "contains_source_text": False,
        },
        "tasks": {},
    }
    total_calls = 0
    total_prompt_chars = 0
    for task in tasks:
        rows = []
        indexed_samples = load_samples(task, args.offset, args.limit)
        for index, sample in indexed_samples:
            gold = normalize_triples(sample.get(TASK_CONFIG[task]["gold_key"]) or [])
            gold_counts = {k: len(v) for k, v in categorize(gold).items()}
            prompts = {
                system: build_prompt(
                    sample,
                    task,
                    system,
                    args.max_context_chars,
                    args.confidence_threshold,
                    args.max_nontax,
                    args.implicit_confidence_threshold,
                )
                for system in systems
            }
            rows.append(
                {
                    "sample_index": index,
                    "sample_id_sha256": _sha256(str(sample.get("id") or index)),
                    "input_fields": (
                        ["context"]
                        if task == "a"
                        else [
                            "context",
                            "initial-primitive-ontology-triples",
                            "terms",
                            "types",
                        ]
                    ),
                    "protected_gold_field": TASK_CONFIG[task]["gold_key"],
                    "context_chars_sent": min(
                        len(str(sample.get("context") or "")), args.max_context_chars
                    ),
                    "initial_triples_sent": (
                        len(sample.get("initial-primitive-ontology-triples") or [])
                        if task == "b"
                        else 0
                    ),
                    "given_terms_sent": len(sample.get("terms") or []) if task == "b" else 0,
                    "given_types_sent": len(sample.get("types") or []) if task == "b" else 0,
                    "gold_counts_scoring_only": gold_counts,
                    "prompts": {
                        system: {
                            "sha256": _sha256(prompt),
                            "chars": len(prompt),
                        }
                        for system, prompt in prompts.items()
                    },
                }
            )
            total_calls += len(systems)
            total_prompt_chars += sum(len(prompt) for prompt in prompts.values())
        manifest["tasks"][f"task_{task}"] = {
            "task_name": TASK_CONFIG[task]["name"],
            "n_samples": len(indexed_samples),
            "estimated_calls": len(indexed_samples) * len(systems),
            "samples": rows,
        }
    manifest["meta"]["estimated_calls"] = total_calls
    manifest["meta"]["total_prompt_chars"] = total_prompt_chars
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1 or args.workers > 4:
        raise SystemExit("--workers must be between 1 and 4")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise SystemExit("--confidence-threshold must be between 0 and 1")
    if not 0.0 <= args.implicit_confidence_threshold <= 1.0:
        raise SystemExit("--implicit-confidence-threshold must be between 0 and 1")
    if not 0.0 <= args.calibrated_confidence_threshold <= 1.0:
        raise SystemExit("--calibrated-confidence-threshold must be between 0 and 1")
    if args.max_nontax < 0:
        raise SystemExit("--max-nontax must be non-negative")
    if args.max_implicit_gap_chars < 0:
        raise SystemExit("--max-implicit-gap-chars must be non-negative")
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = sorted(set(systems) - set(SYSTEMS))
    if unknown:
        raise SystemExit(f"unsupported systems: {', '.join(unknown)}")

    tasks = ["a", "b"] if args.task == "ab" else [args.task]
    if args.audit_only:
        manifest = build_audit_manifest(args, tasks, systems)
        out = RESULTS / f"official_blind_audit_{args.task}_n{args.limit}_off{args.offset}.json"
        atomic_write_json(out, manifest)
        print(
            f"Audit only: calls={manifest['meta']['estimated_calls']} "
            f"prompt_chars={manifest['meta']['total_prompt_chars']} destination="
            f"{manifest['meta']['external_destination']}",
            flush=True,
        )
        print(f"Wrote {out}", flush=True)
        return 0

    any_failed = False
    for task in tasks:
        indexed_samples = load_samples(task, args.offset, args.limit)
        print(
            f"Task {task.upper()} {TASK_CONFIG[task]['name']}: n={len(indexed_samples)} "
            f"offset={args.offset} systems={','.join(systems)} workers={args.workers}",
            flush=True,
        )
        task_payload: dict[str, Any] = {
            "meta": {
                "challenge": "LLMs4OL 2026",
                "protocol": PROTOCOL,
                "task": task,
                "model": args.model,
                "offset": args.offset,
                "limit": args.limit,
                "gold_visible_to_model": False,
            },
            "systems": {},
        }
        for system in systems:
            checkpoint = (
                RESULTS
                / "checkpoints"
                / f"blind_{_slug(args.backend)}_task{task}_{system}_n{args.limit}_off{args.offset}.json"
            )
            payload = run_system(
                args=args,
                task=task,
                system=system,
                indexed_samples=indexed_samples,
                checkpoint_path=checkpoint,
            )
            task_payload["systems"][system] = payload
            summary = payload["summary"]
            any_failed = any_failed or not summary["complete"]
            print(
                f"  {system}: ok={summary['n_success']}/{summary['target_n']} "
                f"TypF1={summary['term_typing_f1_micro']['f1']:.4f} "
                f"TaxF1={summary['taxonomy_f1_micro']['f1']:.4f} "
                f"NonTaxF1={summary['nontax_f1_micro']['f1']:.4f} "
                f"GS={summary['official_graph_similarity_avg']:.4f}",
                flush=True,
            )
        out = RESULTS / f"official_blind_{_slug(args.backend)}_{task}_n{args.limit}_off{args.offset}.json"
        atomic_write_json(out, task_payload)
        print(f"Wrote {out}", flush=True)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
