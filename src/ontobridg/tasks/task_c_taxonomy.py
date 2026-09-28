"""Task C — cross-domain taxonomy discovery (LLMs4OL 2026 Taxonomy task).

Official definition (`competition/OFFICIAL_2026.md` §Task C): given ontologies
from source domains, induce an ``is-a`` hierarchy for an **unseen** domain.
Input at test time is a plain term list; the output is gold-shaped edges
``{"id", "parent", "child"}`` for the submission triple ``(child, is-a, parent)``.

Two offline modes are implemented here (both zero-API, deterministic):

``zero_shot_surface``
    No source-domain training. Parents are induced from in-list containment and
    morphology only.

``source_transfer``
    Source-domain taxonomies are mined into :class:`SourceTaxonomyRules`
    (child->parent, head-token->parent) and transferred to unseen terms.

All candidates pass a symbolic repair stage (self-loop removal, parent cap,
acyclicity) so the emitted graph is a forest by construction.

:class:`OntoLearnerTaxonomyLearner` wraps the system in the OntoLearner
``AutoLearner`` contract (``load`` / ``fit`` / ``predict`` / ``fit_predict``
with ``task="taxonomy-discovery"``) so ``ontology.extract()`` results and
plain held-out term lists both plug straight into the official protocol
(`competition/OFFICIAL_2026.md` §Task C).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ontobridg.eval.official_taxonomy_match import evaluate as evaluate_taxonomy

TASK_C_VERSION = "official_task_c_v1"
DOMAIN_STEMS = ("Archeon", "Biora", "Hylex", "Phyra")
DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "LLMs4OL-Challenge" / "2026"

EDGE_ID_PREFIX = "TR_"


def normalize(text: Any) -> str:
    return " ".join(str(text).lower().strip().split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def challenge_root(root: Path | None = None) -> Path:
    return Path(root) if root is not None else DEFAULT_ROOT


def domain_paths(stem: str, *, root: Path | None = None) -> tuple[Path, Path]:
    base = challenge_root(root) / "TaskC-Taxonomy"
    return base / f"{stem}_test_input.txt", base / f"{stem}_gold.json"


def load_terms(path: Path) -> list[str]:
    """Read a Task C test-input file (one surface term per line)."""
    terms: list[str] = []
    seen: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if not term or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def load_gold_edges(path: Path) -> list[dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    edges: list[dict[str, str]] = []
    for row in payload:
        parent = row.get("parent")
        child = row.get("child")
        if parent is None or child is None:
            continue
        edges.append(
            {
                "id": str(row.get("id") or f"{EDGE_ID_PREFIX}{len(edges):08x}"),
                "parent": str(parent),
                "child": str(child),
            }
        )
    return edges


@dataclass(frozen=True)
class TaskCDomain:
    """One official Task C domain: held-out term list + gold ``is-a`` edges."""

    name: str
    terms: list[str]
    gold_edges: list[dict[str, str]]
    input_sha256: str = ""
    gold_sha256: str = ""
    gold_edge_count_total: int = 0

    def term_set(self) -> set[str]:
        return {normalize(term) for term in self.terms}

    def gold_for_terms(self, terms: Sequence[str] | None = None) -> list[dict[str, str]]:
        wanted = {normalize(term) for term in (terms if terms is not None else self.terms)}
        return [edge for edge in self.gold_edges if normalize(edge["child"]) in wanted]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "terms": list(self.terms),
            "gold_edges": list(self.gold_edges),
            "input_sha256": self.input_sha256,
            "gold_sha256": self.gold_sha256,
            "gold_edge_count_total": self.gold_edge_count_total,
        }


def load_task_c_domain(
    stem: str,
    *,
    root: Path | None = None,
    term_limit: int | None = None,
    with_gold: bool = True,
) -> TaskCDomain:
    """Load one Task C domain, optionally truncated to a deterministic prefix."""
    input_path, gold_path = domain_paths(stem, root=root)
    terms = load_terms(input_path)
    gold_edges: list[dict[str, str]] = []
    gold_total = 0
    if with_gold and gold_path.exists():
        gold_edges = load_gold_edges(gold_path)
        gold_total = len(gold_edges)
    selected = terms[:term_limit] if term_limit else terms
    if with_gold and gold_path.exists() and term_limit:
        gold_edges = TaskCDomain(
            name=stem, terms=selected, gold_edges=gold_edges
        ).gold_for_terms(selected)
    return TaskCDomain(
        name=stem,
        terms=selected,
        gold_edges=gold_edges,
        input_sha256=sha256_file(input_path) if input_path.exists() else "",
        gold_sha256=sha256_file(gold_path) if gold_path.exists() else "",
        gold_edge_count_total=gold_total,
    )


# --------------------------------------------------------------------------- #
# Source-domain transfer rules
# --------------------------------------------------------------------------- #
def edge_pairs(edges: Iterable[Any]) -> list[tuple[str, str]]:
    """Coerce edges to ``(parent, child)`` pairs, keeping first-seen order.

    Accepts dict-like rows (``{"parent", "child"}``), attribute rows such as
    OntoLearner's ``TaxonomicRelation`` (``.parent`` / ``.child``), and
    positional sequences ``(parent, child)``.
    """
    pairs: list[tuple[str, str]] = []
    for row in edges:
        if isinstance(row, Mapping):
            parent = row.get("parent")
            child = row.get("child")
        elif hasattr(row, "parent") and hasattr(row, "child"):
            parent = row.parent
            child = row.child
        elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and len(row) >= 2:
            parent, child = row[0], row[1]
        else:
            continue
        if parent is None or child is None:
            continue
        pairs.append((normalize(parent), normalize(child)))
    return pairs


def head_token(term: str) -> str:
    """Surface head of a term (trailing tokens); the transfer key."""
    tokens = normalize(term).split()
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return " ".join(tokens[-2:])


def singularize(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def rules_sha256(pairs: Sequence[tuple[str, str]]) -> str:
    payload = "\n".join(f"{parent}\t{child}" for parent, child in pairs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class SourceTaxonomyRules:
    """Transferable state mined from source-domain taxonomies."""

    child_parents: dict[str, list[str]] = field(default_factory=dict)
    head_parents: dict[str, list[str]] = field(default_factory=dict)
    parent_frequency: dict[str, int] = field(default_factory=dict)
    n_edges: int = 0
    edges_sha256: str = ""

    def parents_for_child(self, child: str) -> list[str]:
        return list(self.child_parents.get(normalize(child), ()))

    def parents_for_head(self, term: str) -> list[str]:
        return list(self.head_parents.get(head_token(term), ()))

    def to_json(self) -> dict[str, Any]:
        return {
            "child_parents": {k: list(v) for k, v in sorted(self.child_parents.items())},
            "head_parents": {k: list(v) for k, v in sorted(self.head_parents.items())},
            "parent_frequency": dict(sorted(self.parent_frequency.items())),
            "n_edges": int(self.n_edges),
            "edges_sha256": self.edges_sha256,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "SourceTaxonomyRules":
        return cls(
            child_parents={
                str(key): [str(x) for x in value]
                for key, value in (payload.get("child_parents") or {}).items()
            },
            head_parents={
                str(key): [str(x) for x in value]
                for key, value in (payload.get("head_parents") or {}).items()
            },
            parent_frequency={
                str(key): int(value)
                for key, value in (payload.get("parent_frequency") or {}).items()
            },
            n_edges=int(payload.get("n_edges") or 0),
            edges_sha256=str(payload.get("edges_sha256") or ""),
        )


def mine_source_rules(
    edges: Iterable[Any], *, max_parents_per_head: int = 8
) -> SourceTaxonomyRules:
    """Mine child->parent and head-token->parent transfer rules from edges."""
    pairs = edge_pairs(edges)
    rules = SourceTaxonomyRules(n_edges=len(pairs), edges_sha256=rules_sha256(pairs))
    head_counts: dict[str, dict[str, int]] = {}
    for parent, child in pairs:
        rules.child_parents.setdefault(child, [])
        if parent not in rules.child_parents[child]:
            rules.child_parents[child].append(parent)
        rules.parent_frequency[parent] = rules.parent_frequency.get(parent, 0) + 1
        head = head_token(child)
        if head:
            head_counts.setdefault(head, {})
            head_counts[head][parent] = head_counts[head].get(parent, 0) + 1
    for head, counts in head_counts.items():
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        rules.head_parents[head] = [parent for parent, _ in ranked[:max_parents_per_head]]
    return rules




# --------------------------------------------------------------------------- #
# Deterministic surface + transfer inducer (zero-API)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TaxonomyCandidate:
    parent: str
    score: float
    reason: str


MIN_CLAIM_EXACT_F1 = 0.10


class SurfaceTaxonomyInducer:
    """Zero-API taxonomy inducer: surface rules + source transfer + repair.

    Score schedule (documented, deterministic, no learned weights):

    ====================  ======  =========================================
    reason                 score   meaning
    ====================  ======  =========================================
    source_child_match      1.00   child surface seen in a source taxonomy
    suffix_containment      0.80   candidate parent is a strict suffix
    source_head_transfer    0.62   child head token seen in source data
    morphology_singular     0.55   plural child of an in-list singular term
    substring_containment   0.45   candidate parent occurs inside the child
    ====================  ======  =========================================
    """

    def __init__(
        self,
        *,
        max_parents_per_child: int = 1,
        use_source_transfer: bool = True,
        use_containment: bool = True,
        use_morphology: bool = True,
        acyclic_repair: bool = True,
    ) -> None:
        self.max_parents_per_child = max(1, int(max_parents_per_child))
        self.use_source_transfer = use_source_transfer
        self.use_containment = use_containment
        self.use_morphology = use_morphology
        self.acyclic_repair = acyclic_repair
        self.rules = SourceTaxonomyRules()
        self.last_stats: dict[str, Any] = {}

    def fit(self, edges: Iterable[Any]) -> SourceTaxonomyRules:
        self.rules = mine_source_rules(edges)
        return self.rules

    def candidates(
        self, term: str, pool: Sequence[str], rules: SourceTaxonomyRules | None = None
    ) -> list[TaxonomyCandidate]:
        key = normalize(term)
        if not key:
            return []
        active = self.rules if rules is None else rules
        found: list[TaxonomyCandidate] = []
        if self.use_source_transfer and active.n_edges:
            for parent in active.parents_for_child(term):
                if parent and parent != key:
                    found.append(TaxonomyCandidate(parent, 1.0, "source_child_match"))
            if not found:
                for parent in active.parents_for_head(term):
                    if parent and parent != key:
                        found.append(TaxonomyCandidate(parent, 0.62, "source_head_transfer"))
        pool_norm = {normalize(other) for other in pool if normalize(other) != key}
        if self.use_containment:
            for other in sorted(pool_norm):
                if key.endswith(" " + other):
                    found.append(TaxonomyCandidate(other, 0.80, "suffix_containment"))
                elif other in key:
                    found.append(TaxonomyCandidate(other, 0.45, "substring_containment"))
        if self.use_morphology:
            tokens = key.split()
            if tokens:
                singular_head = singularize(tokens[-1])
                if singular_head != tokens[-1]:
                    singular = " ".join(tokens[:-1] + [singular_head])
                    if singular in pool_norm:
                        found.append(
                            TaxonomyCandidate(singular, 0.55, "morphology_singular")
                        )
        best: dict[str, TaxonomyCandidate] = {}
        for candidate in found:
            current = best.get(candidate.parent)
            if current is None or (candidate.score, candidate.reason) > (
                current.score,
                current.reason,
            ):
                best[candidate.parent] = candidate
        return sorted(best.values(), key=lambda item: (-item.score, item.parent))

    def select(
        self, term: str, pool: Sequence[str], rules: SourceTaxonomyRules | None = None
    ) -> list[TaxonomyCandidate]:
        return self.candidates(term, pool, rules)[: self.max_parents_per_child]

    def predict(
        self, terms: Sequence[str], *, rules: SourceTaxonomyRules | None = None
    ) -> list[dict[str, str]]:
        """Induce ``is-a`` edges for a held-out term list (no gold, no API)."""
        pool = [normalize(term) for term in terms]
        selected: list[tuple[str, str, float]] = []
        uncovered: list[str] = []
        for term in terms:
            candidates = self.select(term, pool, rules)
            if not candidates:
                uncovered.append(term)
                continue
            selected.extend(
                (normalize(term), candidate.parent, candidate.score)
                for candidate in candidates
            )
        kept, dropped = self._repair(selected)
        edges = [
            {
                "id": f"{EDGE_ID_PREFIX}{hashlib.sha256(f'{child}|{parent}'.encode('utf-8')).hexdigest()[:8]}",
                "parent": parent,
                "child": child,
            }
            for child, parent, _ in kept
        ]
        self.last_stats = {
            "terms": len(terms),
            "covered_children": len({child for child, _, _ in kept}),
            "uncovered_terms": uncovered,
            "repair_dropped": len(dropped),
            "predicted_edges": len(edges),
        }
        return edges

    def _repair(
        self, selected: Sequence[tuple[str, str, float]]
    ) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
        """Deduplicate/self-loop filter, parent cap, then acyclicity repair."""
        best: dict[tuple[str, str], float] = {}
        for child, parent, score in selected:
            if not child or not parent or parent == child:
                continue
            key = (child, parent)
            if key not in best or score > best[key]:
                best[key] = score
        ordered = sorted(
            ((child, parent, score) for (child, parent), score in best.items()),
            key=lambda item: (-item[2], item[0], item[1]),
        )
        if not self.acyclic_repair:
            return ordered, []
        root: dict[str, str] = {}

        def find(node: str) -> str:
            path: list[str] = []
            while root.get(node, node) != node:
                path.append(node)
                node = root[node]
            for item in path:
                root[item] = node
            return node

        kept: list[tuple[str, str, float]] = []
        dropped: list[tuple[str, str, float]] = []
        for child, parent, score in ordered:
            child_root, parent_root = find(child), find(parent)
            if child_root == parent_root:
                dropped.append((child, parent, score))
                continue
            root[parent_root] = child_root
            kept.append((child, parent, score))
        return kept, dropped


class TaskCTaxonomySystem:
    """Adapter-shaped Task C system: ``fit`` on source domains, ``predict`` on terms."""

    MODES = ("zero_shot_surface", "source_transfer")

    def __init__(
        self,
        *,
        mode: str = "source_transfer",
        inducer: SurfaceTaxonomyInducer | None = None,
        max_parents_per_child: int = 1,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown Task C mode: {mode}")
        self.mode = mode
        self.inducer = inducer or SurfaceTaxonomyInducer(
            use_source_transfer=mode == "source_transfer",
            max_parents_per_child=max_parents_per_child,
        )
        self.rules = SourceTaxonomyRules()

    def fit(self, sources: Mapping[str, Any] | Iterable[Any]) -> SourceTaxonomyRules:
        if self.mode == "zero_shot_surface":
            self.rules = SourceTaxonomyRules()
            return self.rules
        collected: list[Any] = []
        rows = sources.values() if isinstance(sources, Mapping) else sources
        for source in rows:
            collected.extend(
                source.gold_edges if isinstance(source, TaskCDomain) else list(source)
            )
        self.rules = self.inducer.fit(collected)
        return self.rules

    def predict(
        self, terms: Sequence[str] | TaskCDomain | None = None, **kwargs: Any
    ) -> list[dict[str, str]]:
        if isinstance(terms, TaskCDomain):
            terms = terms.terms
        return self.inducer.predict(list(terms or []), rules=self.rules, **kwargs)


# --------------------------------------------------------------------------- #
# OntoLearner-aligned wrapper (AutoLearner contract)
# --------------------------------------------------------------------------- #
TAXONOMY_TASK = "taxonomy-discovery"


def _extract_taxonomies(data: Any) -> list[Any]:
    """Return ``taxonomies`` rows from an OntoLearner ``extract()`` result.

    Mirrors ``AutoLearner.tasks_data_former`` / ``tasks_ground_truth_former``
    access patterns (``data.type_taxonomies.taxonomies``); returns ``[]`` for
    anything that does not carry taxonomy rows.
    """
    scope = getattr(data, "type_taxonomies", None)
    if scope is None and isinstance(data, Mapping):
        scope = data.get("type_taxonomies", None)
    rows = getattr(scope, "taxonomies", None) if scope is not None else None
    if rows is None and isinstance(scope, Mapping):
        rows = scope.get("taxonomies")
    if rows is None:
        rows = getattr(data, "taxonomies", None)
    return list(rows) if rows else []


def _terms_of(data: Any) -> list[str]:
    """Derive the prediction term list, mirroring ``tasks_data_former``.

    For ``taxonomy-discovery`` the official OntoLearner former emits the
    deduplicated union of parent and child surfaces; explicit term lists and
    :class:`TaskCDomain` inputs pass through unchanged.
    """
    if isinstance(data, TaskCDomain):
        return list(data.terms)
    if isinstance(data, Mapping) and "terms" in data:
        return [str(term) for term in data.get("terms") or []]
    rows = _extract_taxonomies(data)
    if rows:
        terms: list[str] = []
        seen: set[str] = set()
        for pair in edge_pairs(rows):
            for term in pair:
                if term and term not in seen:
                    seen.add(term)
                    terms.append(term)
        return terms
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if all(isinstance(term, str) for term in data):
            return list(data)
    raise TypeError(
        "taxonomy-discovery eval data must be a term list, TaskCDomain, mapping "
        "with 'terms', or an OntoLearner extract() result with "
        "type_taxonomies.taxonomies"
    )


def _gold_pairs(data: Any) -> list[tuple[str, str]]:
    """Derive gold ``(parent, child)`` rows for ``fit``, any accepted shape."""
    if isinstance(data, TaskCDomain):
        return edge_pairs(data.gold_edges)
    rows = _extract_taxonomies(data)
    if rows:
        return edge_pairs(rows)
    if isinstance(data, Mapping) and "parent" in data and "child" in data:
        return edge_pairs([data])
    if isinstance(data, Mapping):
        collected: list[tuple[str, str]] = []
        for value in data.values():
            if isinstance(value, TaskCDomain):
                collected.extend(edge_pairs(value.gold_edges))
            elif value is not None:
                collected.extend(edge_pairs(value))
        if collected:
            return collected
        raise TypeError("mapping sources must contain TaskCDomain values or edge rows")
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return edge_pairs(data)
    raise TypeError(
        "taxonomy-discovery train data must be a TaskCDomain, a mapping of "
        "domains, edge rows, or an OntoLearner extract() result with "
        "type_taxonomies.taxonomies"
    )


class OntoLearnerTaxonomyLearner:
    """Adapter exposing :class:`TaskCTaxonomySystem` through OntoLearner's
    ``AutoLearner`` contract (``data/OntoLearner/ontolearner/base/learner.py``).

    Signatures match ``AutoLearner`` so the wrapper drops into
    ``LearnerPipeline``-style drivers without the heavyweight ``ontolearner``
    import (torch/transformers are never touched):

    * ``load(**kwargs)`` — records configuration (nothing to load: the
      inducer is deterministic and API-free).
    * ``fit(train_data, task, ontologizer=True)`` — mines transfer rules from
      gold ``is-a`` edges; for extract()-shaped input rows are taken with
      ground-truth-former semantics (``{"parent", "child"}`` pairs), because
      the rules are mined from edges, not bare terms.
    * ``predict(eval_data, task, ontologizer=True)`` — emits gold-shaped
      ``{"id", "parent", "child"}`` edges for a held-out term list.
    * ``fit_predict(train_data, eval_data, task)`` — convenience composition.

    ``ontologizer`` is accepted for signature parity; coercion is driven by
    input shape, so extract() results and already-formatted rows are both
    accepted. Any task other than ``taxonomy-discovery`` raises the same
    ``ValueError`` message as ``AutoLearner.fit``.
    """

    def __init__(
        self,
        *,
        mode: str = "source_transfer",
        max_parents_per_child: int = 1,
        **kwargs: Any,
    ) -> None:
        self.mode = mode
        self.max_parents_per_child = max_parents_per_child
        self.config: dict[str, Any] = dict(kwargs)
        self.system = TaskCTaxonomySystem(
            mode=mode, max_parents_per_child=max_parents_per_child
        )
        self.last_stats: dict[str, Any] = {}

    @staticmethod
    def _check_task(task: str) -> None:
        if task != TAXONOMY_TASK:
            raise ValueError(f"{task} is not a valid task.")

    def load(self, **kwargs: Any) -> None:
        """Record configuration; the deterministic inducer needs no weights."""
        self.config.update(kwargs)

    def fit(
        self, train_data: Any, task: str, ontologizer: bool = True
    ) -> SourceTaxonomyRules:
        self._check_task(task)
        pairs = _gold_pairs(train_data)
        # TaskCTaxonomySystem.fit iterates sources, so the flat pair list is
        # wrapped as a single source batch of (parent, child) rows.
        self.system.fit([pairs])
        return self.system.rules

    def predict(
        self, eval_data: Any, task: str, ontologizer: bool = True
    ) -> list[dict[str, str]]:
        self._check_task(task)
        edges = self.system.predict(_terms_of(eval_data))
        self.last_stats = dict(self.system.inducer.last_stats)
        return edges

    def fit_predict(
        self, train_data: Any, eval_data: Any, task: str
    ) -> list[dict[str, str]]:
        self.fit(train_data, task)
        return self.predict(eval_data, task)



# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def deployment_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Honest gate: a Task C number is only claimable above the trivial floor.

    Even when the numeric floor is met the gate stays closed unless the run used
    the official protocol (external OntoLearner train data). The locally
    reproducible setting is a *surrogate*, so it can never unlock a claim.
    """
    micro = metrics.get("micro", {})
    exact = float(micro.get("exact_match", {}).get("f1_score", 0.0))
    fuzzy = float(micro.get("fuzzy_match", {}).get("f1_score", 0.0))
    fidelity = str(metrics.get("protocol_fidelity") or "local_surrogate")
    reasons: list[str] = []
    if exact < MIN_CLAIM_EXACT_F1:
        reasons.append(f"micro_exact_f1_below_{MIN_CLAIM_EXACT_F1}")
    empty = sorted(
        name
        for name, row in (metrics.get("domains") or {}).items()
        if not row.get("predicted_edges")
    )
    if empty:
        reasons.append("domains_with_zero_predictions:" + ",".join(empty))
    if fidelity != "official":
        reasons.append(f"protocol_fidelity_{fidelity}")
    passed = not reasons
    return {
        "min_claim_exact_f1": MIN_CLAIM_EXACT_F1,
        "observed_exact_f1": exact,
        "observed_fuzzy_f1": fuzzy,
        "protocol_fidelity": fidelity,
        "passed": passed,
        "scientific_deployable": passed,
        "confirmatory_eligible": False,
        "external_reportable": False,
        "reasons": reasons,
        "interpretation": "local_offline_surrogate_not_official_submission",
    }


def evaluate_task_c(
    domains: Mapping[str, TaskCDomain],
    *,
    mode: str = "source_transfer",
    source_domains: Mapping[str, TaskCDomain] | None = None,
    rules: SourceTaxonomyRules | None = None,
    fuzzy_threshold: float = 0.90,
    max_parents_per_child: int = 1,
    include_semantic: bool = False,
) -> dict[str, Any]:
    """Score one Task C mode over the held-out domains, fully offline."""
    if mode == "source_transfer" and rules is None and source_domains is None:
        raise ValueError("source_transfer needs source_domains or fitted rules")
    per_domain: dict[str, Any] = {}
    y_true_all: list[dict[str, str]] = []
    y_pred_all: list[dict[str, str]] = []
    for name, domain in domains.items():
        system = TaskCTaxonomySystem(mode=mode, max_parents_per_child=max_parents_per_child)
        if mode == "source_transfer":
            if rules is not None:
                system.rules = rules
            else:
                sources = {
                    key: value
                    for key, value in (source_domains or {}).items()
                    if key != name
                }
                system.fit(sources)
        predictions = system.predict(domain.terms)
        report = evaluate_taxonomy(
            domain.gold_edges,
            predictions,
            fuzzy_threshold=fuzzy_threshold,
            semantic=include_semantic,
        )
        per_domain[name] = {
            "terms": len(domain.terms),
            "gold_edges": len(domain.gold_edges),
            "predicted_edges": len(predictions),
            "gold_edges_total_in_official_file": domain.gold_edge_count_total,
            "input_sha256": domain.input_sha256,
            "gold_sha256": domain.gold_sha256,
            "exact_match": report["exact_match"],
            "fuzzy_match": report["fuzzy_match"],
            "coverage": (
                len({edge["child"] for edge in predictions}) / len(domain.terms)
                if domain.terms
                else 0.0
            ),
            "stats": dict(system.inducer.last_stats),
            "source_rules": {
                "n_edges": system.rules.n_edges,
                "edges_sha256": system.rules.edges_sha256,
            },
        }
        y_true_all.extend(domain.gold_edges)
        y_pred_all.extend(predictions)
    micro = evaluate_taxonomy(
        y_true_all, y_pred_all, fuzzy_threshold=fuzzy_threshold, semantic=include_semantic
    )
    macro_f1 = (
        sum(row["exact_match"]["f1_score"] for row in per_domain.values()) / len(per_domain)
        if per_domain
        else 0.0
    )
    report: dict[str, Any] = {
        "protocol": TASK_C_VERSION,
        "mode": mode,
        "domains": per_domain,
        "micro": {
            "gold_edges": len(y_true_all),
            "predicted_edges": len(y_pred_all),
            "exact_match": micro["exact_match"],
            "fuzzy_match": micro["fuzzy_match"],
        },
        "macro_exact_f1": macro_f1,
        "fuzzy_threshold": fuzzy_threshold,
        "metric_engines": micro.get("metric_engines", {}),
        "metric_id": "llms4ol2026_task_c_taxonomy",
    }
    report["gate"] = deployment_gate(report)
    report["external_reportable"] = False
    report["scientific_deployable"] = report["gate"]["scientific_deployable"]
    if include_semantic and "semantic_match" in micro:
        report["micro"]["semantic_match"] = micro["semantic_match"]
    return report

