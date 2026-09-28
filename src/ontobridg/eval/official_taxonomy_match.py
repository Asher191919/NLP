"""Official LLMs4OL 2026 Task C (taxonomy) metric port.

Source of truth:
  D:\\NLP\\data\\LLMs4OL-Challenge\\2026\\metrics\\taxonomy.py

Mirrors that module:

- ``exact_match``  鈥?set-based edge precision/recall/F1
- ``fuzzy_match``  鈥?edge similarity + optimal assignment, threshold 0.90
- ``semantic_match`` 鈥?same as fuzzy but with sentence embeddings

POLICY
------
- ``exact_match`` / ``fuzzy_match`` are offline and deterministic; they are the
  only two metrics this project reports for Task C.
- ``semantic_match`` requires the official ``nomic-ai/nomic-embed-text-v1.5``
  sentence-transformer. It is deliberately opt-in (``embedder`` or
  ``allow_download=True``) so no evaluation path performs a hidden network call.

ENGINES
-------
The official module imports ``rapidfuzz`` and ``scipy``. Both are optional here:
a pure-python ``difflib``-based ratio and a deterministic greedy assignment are
used when they are unavailable. ``metric_engines()`` reports which pair was
used, and every report written by this project carries those names.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

OFFICIAL_SOURCE = "data/LLMs4OL-Challenge/2026/metrics/taxonomy.py"
METRIC_ID = "llms4ol2026_task_c_taxonomy"
EXTERNAL_REPORTABLE = True
DEFAULT_FUZZY_THRESHOLD = 0.90

__all__ = [
    "DEFAULT_FUZZY_THRESHOLD",
    "EXTERNAL_REPORTABLE",
    "METRIC_ID",
    "OFFICIAL_SOURCE",
    "SemanticMatchUnavailable",
    "edge_similarity",
    "evaluate",
    "exact_match",
    "fuzzy_match",
    "metric_engines",
    "normalize",
    "semantic_match",
]


class SemanticMatchUnavailable(RuntimeError):
    """Raised when semantic_match is requested without an explicit embedder."""


def metric_engines() -> dict[str, str]:
    """Report which offline engines back the string/assignment operations."""
    try:  # pragma: no cover - depends on environment
        import rapidfuzz  # noqa: F401

        string_engine = "rapidfuzz"
    except Exception:
        string_engine = "difflib"
    try:  # pragma: no cover - depends on environment
        import scipy  # noqa: F401

        assignment_engine = "scipy_hungarian"
    except Exception:
        assignment_engine = "greedy_descent"
    return {"string_similarity": string_engine, "assignment": assignment_engine}


def normalize(text: Any) -> str:
    return " ".join(str(text).lower().strip().split())


def _ratio(left: str, right: str) -> float:
    """Return a 0..1 similarity, matching ``rapidfuzz.fuzz.ratio`` scale."""
    try:  # pragma: no cover - depends on environment
        from rapidfuzz import fuzz

        return fuzz.ratio(left, right) / 100.0
    except Exception:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, left, right).ratio()


def edge_similarity(pred_edge: Sequence[str], gold_edge: Sequence[str]) -> float:
    parent = _ratio(pred_edge[0], gold_edge[0])
    child = _ratio(pred_edge[1], gold_edge[1])
    return (parent + child) / 2.0


def _edges(rows: Iterable[Mapping[str, Any]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        parent = row.get("parent")
        child = row.get("child")
        if parent is None or child is None:
            continue
        out.append((normalize(parent), normalize(child)))
    return out


def _prf(tp: float, n_pred: int, n_gold: int) -> dict[str, float | int]:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"f1_score": f1, "precision": precision, "recall": recall}


def exact_match(
    y_true: Sequence[Mapping[str, Any]], y_pred: Sequence[Mapping[str, Any]]
) -> dict[str, float | int]:
    gold = set(_edges(y_true))
    pred = set(_edges(y_pred))
    return _prf(float(len(gold & pred)), len(pred), len(gold))


def _similarity_matrix(
    pred: list[tuple[str, str]],
    gold: list[tuple[str, str]],
    similarity: Callable[[Sequence[str], Sequence[str]], float],
) -> list[list[float]]:
    return [[similarity(p, g) for g in gold] for p in pred]


def _assign_offline(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Optimal assignment via scipy, else a deterministic greedy fallback."""
    if not cost:
        return []
    try:  # pragma: no cover - depends on environment
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        dense = np.asarray(cost, dtype=float)
        rows, cols = linear_sum_assignment(dense)
        return [(int(r), int(c)) for r, c in zip(rows, cols)]
    except Exception:
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        pairs: list[tuple[int, int]] = []
        cells = sorted(
            ((cost[r][c], r, c) for r in range(len(cost)) for c in range(len(cost[r]))),
            key=lambda item: (item[0], item[1], item[2]),
        )
        for _, row, col in cells:
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(row)
            used_cols.add(col)
            pairs.append((row, col))
            if len(used_rows) == len(cost) or len(used_cols) == len(cost[0]):
                break
        return sorted(pairs)


def fuzzy_match(
    y_true: Sequence[Mapping[str, Any]],
    y_pred: Sequence[Mapping[str, Any]],
    threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> dict[str, float | int]:
    gold = _edges(y_true)
    pred = _edges(y_pred)
    if not gold or not pred:
        return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}
    matrix = _similarity_matrix(pred, gold, edge_similarity)
    cost = [[1.0 - value for value in row] for row in matrix]
    tp = 0.0
    for row, col in _assign_offline(cost):
        similarity = 1.0 - cost[row][col]
        if similarity >= threshold:
            tp += similarity
    return _prf(tp, len(pred), len(gold))


def _dot(left: Any, right: Any) -> float:
    value = float(sum(a * b for a, b in zip(left, right)))
    return max(0.0, min(1.0, value))


def semantic_match(
    y_true: Sequence[Mapping[str, Any]],
    y_pred: Sequence[Mapping[str, Any]],
    threshold: float = DEFAULT_FUZZY_THRESHOLD,
    *,
    embedder: Callable[[list[str]], Any] | None = None,
    allow_download: bool = False,
) -> dict[str, Any]:
    """Official semantic match, with an explicit embedder and no hidden calls."""
    gold = _edges(y_true)
    pred = _edges(y_pred)
    if not gold or not pred:
        return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}
    if embedder is None:
        if not allow_download:
            raise SemanticMatchUnavailable(
                "semantic_match requires an explicit embedder; the official model "
                "nomic-ai/nomic-embed-text-v1.5 needs a network download. Pass "
                "embedder=... or allow_download=True to opt in."
            )
        embedder = _official_embedder()
    terms = sorted({term for edge in gold + pred for term in edge})
    vectors = embedder(terms)
    index = {term: vectors[i] for i, term in enumerate(terms)}

    def similarity(pred_edge: Sequence[str], gold_edge: Sequence[str]) -> float:
        parent = _dot(index[pred_edge[0]], index[gold_edge[0]])
        child = _dot(index[pred_edge[1]], index[gold_edge[1]])
        return (parent + child) / 2.0

    matrix = _similarity_matrix(pred, gold, similarity)
    cost = [[1.0 - value for value in row] for row in matrix]
    tp = 0.0
    for row, col in _assign_offline(cost):
        value = 1.0 - cost[row][col]
        if value >= threshold:
            tp += value
    report: dict[str, Any] = _prf(tp, len(pred), len(gold))
    report["engine"] = "sentence_transformers_nomic"
    return report


def _official_embedder() -> Callable[[list[str]], Any]:  # pragma: no cover - network
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    def embed(terms: list[str]):
        vectors = model.encode(
            terms, convert_to_numpy=True, normalize_embeddings=True, batch_size=2048
        )
        return [list(map(float, row)) for row in vectors]

    return embed


def evaluate(
    y_true: Sequence[Mapping[str, Any]],
    y_pred: Sequence[Mapping[str, Any]],
    *,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    semantic: bool = False,
    embedder: Callable[[list[str]], Any] | None = None,
) -> dict[str, Any]:
    """Official Task C report: exact + fuzzy (+ semantic when requested)."""
    report: dict[str, Any] = {
        "exact_match": exact_match(y_true, y_pred),
        "fuzzy_match": fuzzy_match(y_true, y_pred, threshold=fuzzy_threshold),
        "fuzzy_threshold": fuzzy_threshold,
        "metric_engines": metric_engines(),
    }
    if semantic:
        try:
            report["semantic_match"] = semantic_match(
                y_true, y_pred, threshold=fuzzy_threshold, embedder=embedder
            )
        except SemanticMatchUnavailable as error:
            report["semantic_match"] = {"status": "unavailable", "reason": str(error)}
    return report
