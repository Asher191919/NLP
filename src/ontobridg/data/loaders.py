from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from ontobridg.data.schema import (
    OntologyDataset,
    RelationSample,
    TaxonomySample,
    TermSample,
)


def _open_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_csv_terms(path: str | Path) -> list[TermSample]:
    samples: list[TermSample] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                TermSample(
                    term=row["term"],
                    type_label=row["type"],
                    context=row.get("context") or None,
                )
            )
    return samples


def load_csv_taxonomy(path: str | Path) -> list[TaxonomySample]:
    samples: list[TaxonomySample] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(TaxonomySample(child=row["child"], parent=row["parent"]))
    return samples


def load_csv_relations(path: str | Path) -> list[RelationSample]:
    samples: list[RelationSample] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                RelationSample(head=row["head"], relation=row["relation"], tail=row["tail"])
            )
    return samples


def load_json_ontology(path: str | Path) -> OntologyDataset:
    raw = json.loads(_open_text(path))
    terms = [
        TermSample(
            term=t["term"],
            type_label=t["type"],
            context=t.get("context"),
        )
        for t in raw.get("terms", [])
    ]
    taxonomy = [
        TaxonomySample(child=e["child"], parent=e["parent"])
        for e in raw.get("taxonomy", [])
    ]
    relations = [
        RelationSample(head=r["head"], relation=r["relation"], tail=r["tail"])
        for r in raw.get("relations", [])
    ]
    domain_ranges = {
        k: (v[0], v[1]) for k, v in raw.get("domain_ranges", {}).items()
    }
    return OntologyDataset(
        name=raw.get("name", Path(path).stem),
        types=list(raw.get("types", [])),
        terms=terms,
        taxonomy=taxonomy,
        relations=relations,
        domain_ranges=domain_ranges,
        metadata=raw.get("metadata", {}),
    )


def save_json_ontology(dataset: OntologyDataset, path: str | Path) -> None:
    payload = {
        "name": dataset.name,
        "types": dataset.types,
        "terms": [
            {"term": t.term, "type": t.type_label, "context": t.context}
            for t in dataset.terms
        ],
        "taxonomy": [{"child": e.child, "parent": e.parent} for e in dataset.taxonomy],
        "relations": [
            {"head": r.head, "relation": r.relation, "tail": r.tail}
            for r in dataset.relations
        ],
        "domain_ranges": {k: list(v) for k, v in dataset.domain_ranges.items()},
        "metadata": dataset.metadata,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_competition_folder(folder: str | Path) -> OntologyDataset:
    folder_path = Path(folder)
    terms = load_csv_terms(folder_path / "terms.csv")
    taxonomy = load_csv_taxonomy(folder_path / "taxonomy.csv")
    rel_path = folder_path / "relations.csv"
    relations = load_csv_relations(rel_path) if rel_path.exists() else []
    types = sorted({t.type_label for t in terms} | {e.child for e in taxonomy} | {e.parent for e in taxonomy})
    return OntologyDataset(
        name=folder_path.name,
        types=types,
        terms=terms,
        taxonomy=taxonomy,
        relations=relations,
    )


def iter_type_pairs(types: Iterable[str], known_edges: set[tuple[str, str]]) -> list[tuple[str, str]]:
    type_list = list(types)
    pairs: list[tuple[str, str]] = []
    for a in type_list:
        for b in type_list:
            if a == b:
                continue
            if (a, b) in known_edges:
                pairs.append((a, b))
            elif (b, a) not in known_edges:
                pairs.append((a, b))
    return pairs
