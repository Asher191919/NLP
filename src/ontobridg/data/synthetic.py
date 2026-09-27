"""Multi-domain synthetic ontology corpora for LLMs4OL-style simulation.

Covers biomedical-ish, geo-ish, and product-ish domains with terms, types,
is-a edges, and non-taxonomic relations (part-of, causes, located-in, treats...).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ontobridg.data.schema import (
    OntologyDataset,
    OntologyKnowledge,
    RelationSample,
    TaxonomySample,
    TermSample,
)


@dataclass
class DomainSpec:
    name: str
    types: dict[str, str]  # type -> parent type (or "" for root)
    terms: list[tuple[str, str]]  # (term, type)
    # (head_term, relation, tail_term)
    relations: list[tuple[str, str, str]]
    # relation -> (domain_type, range_type)
    domain_ranges: dict[str, tuple[str, str]]
    # lexical patterns that appear in "text" for open-vocab mining
    surface_patterns: dict[str, list[str]] = field(default_factory=dict)


BIOMEDICAL = DomainSpec(
    name="biomedical",
    types={
        "Entity": "",
        "BiologicalEntity": "Entity",
        "MedicalEntity": "Entity",
        "Disease": "MedicalEntity",
        "Drug": "MedicalEntity",
        "Symptom": "MedicalEntity",
        "Protein": "BiologicalEntity",
        "Anatomy": "BiologicalEntity",
        "Pathway": "BiologicalEntity",
    },
    terms=[
        ("diabetes", "Disease"),
        ("hypertension", "Disease"),
        ("asthma", "Disease"),
        ("alzheimer", "Disease"),
        ("metformin", "Drug"),
        ("insulin", "Drug"),
        ("lisinopril", "Drug"),
        ("albuterol", "Drug"),
        ("fatigue", "Symptom"),
        ("headache", "Symptom"),
        ("wheezing", "Symptom"),
        ("memory_loss", "Symptom"),
        ("insulin_receptor", "Protein"),
        ("amyloid_beta", "Protein"),
        ("ACE2", "Protein"),
        ("lung", "Anatomy"),
        ("pancreas", "Anatomy"),
        ("brain", "Anatomy"),
        ("glycolysis", "Pathway"),
        ("insulin_signaling", "Pathway"),
    ],
    relations=[
        ("metformin", "treats", "diabetes"),
        ("insulin", "treats", "diabetes"),
        ("lisinopril", "treats", "hypertension"),
        ("albuterol", "treats", "asthma"),
        ("diabetes", "causes", "fatigue"),
        ("hypertension", "causes", "headache"),
        ("asthma", "causes", "wheezing"),
        ("alzheimer", "causes", "memory_loss"),
        ("pancreas", "part-of", "biological_system"),
        ("lung", "part-of", "respiratory_system"),
        ("brain", "part-of", "nervous_system"),
        ("insulin_receptor", "part-of", "insulin_signaling"),
        ("amyloid_beta", "causes", "alzheimer"),
        ("insulin_receptor", "interacts-with", "insulin"),
        ("diabetes", "located-in", "pancreas"),
        ("asthma", "located-in", "lung"),
        ("alzheimer", "located-in", "brain"),
        ("glycolysis", "part-of", "cellular_metabolism"),
        ("metformin", "interacts-with", "insulin_signaling"),
        ("insulin", "interacts-with", "insulin_receptor"),
    ],
    domain_ranges={
        "treats": ("Drug", "Disease"),
        "causes": ("Disease", "Symptom"),
        "part-of": ("BiologicalEntity", "BiologicalEntity"),
        "interacts-with": ("Protein", "Protein"),
        "located-in": ("Disease", "Anatomy"),
        "participates-in": ("Protein", "Pathway"),
    },
    surface_patterns={
        "treats": ["{h} is used to treat {t}", "{h} treats {t}", "{h} is a therapy for {t}"],
        "causes": ["{h} causes {t}", "{h} leads to {t}", "{h} results in {t}"],
        "part-of": ["{h} is part of {t}", "{h} is a component of {t}"],
        "interacts-with": ["{h} interacts with {t}", "{h} binds {t}"],
        "located-in": ["{h} is located in {t}", "{h} affects the {t}"],
    },
)

GEO = DomainSpec(
    name="geo",
    types={
        "Entity": "",
        "Place": "Entity",
        "PoliticalUnit": "Place",
        "NaturalFeature": "Place",
        "City": "PoliticalUnit",
        "Country": "PoliticalUnit",
        "River": "NaturalFeature",
        "Mountain": "NaturalFeature",
        "Continent": "Place",
    },
    terms=[
        ("paris", "City"),
        ("tokyo", "City"),
        ("cairo", "City"),
        ("france", "Country"),
        ("japan", "Country"),
        ("egypt", "Country"),
        ("nile", "River"),
        ("seine", "River"),
        ("amazon_river", "River"),
        ("fuji", "Mountain"),
        ("alps", "Mountain"),
        ("andes", "Mountain"),
        ("europe", "Continent"),
        ("asia", "Continent"),
        ("africa", "Continent"),
        ("south_america", "Continent"),
    ],
    relations=[
        ("paris", "located-in", "france"),
        ("tokyo", "located-in", "japan"),
        ("cairo", "located-in", "egypt"),
        ("france", "located-in", "europe"),
        ("japan", "located-in", "asia"),
        ("egypt", "located-in", "africa"),
        ("nile", "flows-through", "egypt"),
        ("seine", "flows-through", "france"),
        ("paris", "capital-of", "france"),
        ("tokyo", "capital-of", "japan"),
        ("cairo", "capital-of", "egypt"),
        ("alps", "located-in", "europe"),
        ("fuji", "located-in", "japan"),
        ("andes", "located-in", "south_america"),
        ("france", "borders", "alps"),
        ("amazon_river", "located-in", "south_america"),
        ("nile", "part-of", "african_hydrology"),
        ("seine", "part-of", "european_hydrology"),
    ],
    domain_ranges={
        "located-in": ("Place", "Place"),
        "flows-through": ("River", "PoliticalUnit"),
        "capital-of": ("City", "Country"),
        "borders": ("Country", "NaturalFeature"),
        "part-of": ("NaturalFeature", "Entity"),
    },
    surface_patterns={
        "located-in": ["{h} is located in {t}", "{h} lies in {t}", "{h} is in {t}"],
        "flows-through": ["{h} flows through {t}", "{h} runs through {t}"],
        "capital-of": ["{h} is the capital of {t}", "{h} is capital city of {t}"],
        "borders": ["{h} borders {t}", "{h} is adjacent to {t}"],
        "part-of": ["{h} is part of {t}"],
    },
)

PRODUCT = DomainSpec(
    name="product",
    types={
        "Entity": "",
        "CommercialEntity": "Entity",
        "Product": "CommercialEntity",
        "Brand": "CommercialEntity",
        "Component": "Entity",
        "Material": "Component",
        "Feature": "Entity",
        "Category": "CommercialEntity",
        "Device": "Product",
        "Appliance": "Product",
    },
    terms=[
        ("iphone_15", "Device"),
        ("galaxy_s24", "Device"),
        ("pixel_8", "Device"),
        ("dyson_v15", "Appliance"),
        ("roomba_j7", "Appliance"),
        ("apple", "Brand"),
        ("samsung", "Brand"),
        ("google", "Brand"),
        ("dyson", "Brand"),
        ("irobot", "Brand"),
        ("lithium", "Material"),
        ("aluminum", "Material"),
        ("glass", "Material"),
        ("oled_display", "Feature"),
        ("fast_charging", "Feature"),
        ("hepa_filter", "Feature"),
        ("smartphone", "Category"),
        ("vacuum", "Category"),
    ],
    relations=[
        ("iphone_15", "manufactured-by", "apple"),
        ("galaxy_s24", "manufactured-by", "samsung"),
        ("pixel_8", "manufactured-by", "google"),
        ("dyson_v15", "manufactured-by", "dyson"),
        ("roomba_j7", "manufactured-by", "irobot"),
        ("iphone_15", "made-of", "aluminum"),
        ("iphone_15", "made-of", "glass"),
        ("galaxy_s24", "made-of", "aluminum"),
        ("iphone_15", "has-feature", "oled_display"),
        ("iphone_15", "has-feature", "fast_charging"),
        ("galaxy_s24", "has-feature", "oled_display"),
        ("pixel_8", "has-feature", "oled_display"),
        ("dyson_v15", "has-feature", "hepa_filter"),
        ("dyson_v15", "made-of", "aluminum"),
        ("iphone_15", "belongs-to", "smartphone"),
        ("galaxy_s24", "belongs-to", "smartphone"),
        ("pixel_8", "belongs-to", "smartphone"),
        ("dyson_v15", "belongs-to", "vacuum"),
        ("roomba_j7", "belongs-to", "vacuum"),
        ("lithium", "part-of", "battery_pack"),
        ("pixel_8", "manufactured-by", "google"),
    ],
    domain_ranges={
        "manufactured-by": ("Product", "Brand"),
        "made-of": ("Product", "Material"),
        "has-feature": ("Product", "Feature"),
        "belongs-to": ("Product", "Category"),
        "part-of": ("Material", "Entity"),
    },
    surface_patterns={
        "manufactured-by": ["{h} is manufactured by {t}", "{h} is made by {t}", "{h} is a {t} product"],
        "made-of": ["{h} is made of {t}", "{h} uses {t}"],
        "has-feature": ["{h} has feature {t}", "{h} includes {t}"],
        "belongs-to": ["{h} belongs to {t} category", "{h} is a {t}"],
        "part-of": ["{h} is part of {t}"],
    },
)


ALL_DOMAINS = [BIOMEDICAL, GEO, PRODUCT]


def _render_corpus(domain: DomainSpec, rng: random.Random) -> list[str]:
    """Render surface-pattern sentences that encode non-tax relations."""
    sentences: list[str] = []
    term_set = {t for t, _ in domain.terms}
    for h, rel, t in domain.relations:
        patterns = domain.surface_patterns.get(rel)
        if not patterns:
            patterns = [f"{h} {{rel}} {t}".replace("{rel}", rel)]
        pattern = patterns[rng.randrange(len(patterns))]
        # only emit if both ends look like known terms or close variants
        if h in term_set or t in term_set:
            sentences.append(pattern.format(h=h, t=t))
    # shuffle + light noise
    noise = [
        "This document discusses ontology learning from domain text.",
        "Several entities appear in the knowledge base.",
        "Evaluation uses semantic graph similarity and F1.",
    ]
    sentences.extend(noise)
    rng.shuffle(sentences)
    return sentences


def make_synthetic_dataset(
    domain: str | DomainSpec = "biomedical",
    *,
    holdout_fraction: float = 0.0,
    seed: int = 42,
    with_corpus: bool = True,
) -> OntologyDataset:
    """Build an OntologyDataset from a DomainSpec.

    If holdout_fraction > 0, that fraction of terms is removed from the
    returned dataset (kept only in metadata['heldout']) for contamination tests.
    """
    if isinstance(domain, str):
        specs = {d.name: d for d in ALL_DOMAINS}
        if domain == "all":
            return make_multi_domain_dataset(seed=seed, holdout_fraction=holdout_fraction)
        if domain not in specs:
            raise ValueError(f"Unknown domain {domain!r}; choose from {list(specs)}")
        spec = specs[domain]
    else:
        spec = domain

    rng = random.Random(seed)
    terms = [TermSample(term=t, type_label=ty) for t, ty in spec.terms]

    # taxonomy: type-level hierarchy + term-instance is-a edges (term -> its type)
    taxonomy: list[TaxonomySample] = []
    for child, parent in spec.types.items():
        if parent:
            taxonomy.append(TaxonomySample(child=child, parent=parent))
    for term_name, type_name in spec.terms:
        taxonomy.append(TaxonomySample(child=term_name, parent=type_name))

    relations = [RelationSample(head=h, relation=r, tail=t) for h, r, t in spec.relations]
    types = sorted(set(spec.types.keys()) | {ty for _, ty in spec.terms})

    corpus = _render_corpus(spec, rng) if with_corpus else []

    heldout: list[TermSample] = []
    if holdout_fraction > 0:
        n_hold = max(1, int(len(terms) * holdout_fraction))
        idx = list(range(len(terms)))
        rng.shuffle(idx)
        hold_idx = set(idx[:n_hold])
        heldout = [terms[i] for i in hold_idx]
        terms = [terms[i] for i in range(len(terms)) if i not in hold_idx]
        kept = {t.term for t in terms}
        taxonomy = [
            TaxonomySample(child=e.child, parent=e.parent)
            for e in taxonomy
            if e.child in spec.types  # type-level taxonomy always kept
            or (e.child in kept and e.parent in kept)
        ]
        relations = [
            r for r in relations if r.head in kept and r.tail in kept
        ]

    ds = OntologyDataset(
        name=f"synthetic_{spec.name}",
        types=types,
        terms=terms,
        taxonomy=taxonomy,
        relations=relations,
        domain_ranges=dict(spec.domain_ranges),
        metadata={
            "domain": spec.name,
            "corpus": corpus,
            "heldout": [
                {"term": h.term, "type": h.type_label} for h in heldout
            ],
            "seed": seed,
        },
    )
    return ds


def make_multi_domain_dataset(*, seed: int = 42, holdout_fraction: float = 0.0) -> OntologyDataset:
    """Merge all three domains into one multi-domain synthetic corpus."""
    parts = [
        make_synthetic_dataset(d, seed=seed + i, holdout_fraction=holdout_fraction)
        for i, d in enumerate(ALL_DOMAINS)
    ]
    types: list[str] = []
    terms: list[TermSample] = []
    taxonomy: list[TaxonomySample] = []
    relations: list[RelationSample] = []
    domain_ranges: dict[str, tuple[str, str]] = {}
    corpus: list[str] = []
    heldout: list[dict] = []
    for p in parts:
        types.extend(p.types)
        terms.extend(p.terms)
        taxonomy.extend(p.taxonomy)
        relations.extend(p.relations)
        # merge domain-range: keep first non-conflicting; conflicts become multi
        for k, v in p.domain_ranges.items():
            if k not in domain_ranges:
                domain_ranges[k] = v
        corpus.extend(p.metadata.get("corpus", []))
        heldout.extend(p.metadata.get("heldout", []))
    random.Random(seed).shuffle(corpus)
    return OntologyDataset(
        name="synthetic_multi_domain",
        types=sorted(set(types)),
        terms=terms,
        taxonomy=taxonomy,
        relations=relations,
        domain_ranges=domain_ranges,
        metadata={
            "domain": "multi",
            "corpus": corpus,
            "heldout": heldout,
            "seed": seed,
        },
    )


def default_knowledge_from_dataset(
    dataset: OntologyDataset,
    *,
    contamination: float = 0.65,
    seed: int = 0,
) -> OntologyKnowledge:
    """Simulate LLM pretraining knowledge: a *partial* view of the gold ontology.

    contamination controls how much of gold is memorized (term->type, parents,
    relations). Low contamination => novel-domain regime.
    """
    rng = random.Random(seed)
    know = OntologyKnowledge(domain_ranges=dict(dataset.domain_ranges))
    for t in dataset.terms:
        if rng.random() < contamination:
            know.term_to_type[t.term] = t.type_label
        lex = know.type_lexicon.setdefault(t.type_label, [])
        if t.term not in lex:
            lex.append(t.term)
    for e in dataset.taxonomy:
        if rng.random() < max(contamination, 0.4):  # type hierarchy more stable
            know.parents.setdefault(e.child, set()).add(e.parent)
    for r in dataset.relations:
        if rng.random() < contamination:
            know.relations.add(r.triple)
    # always know type-level domain/range structure (symbolic priors)
    for k, v in dataset.domain_ranges.items():
        know.domain_ranges[k] = v
    return know


def write_sample_csvs(dataset: OntologyDataset, out_dir: str) -> None:
    """Write terms/taxonomy/relations CSVs matching LLMs4OL-style schemas."""
    import csv
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "terms.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["term", "type", "context"])
        w.writeheader()
        for t in dataset.terms:
            w.writerow({"term": t.term, "type": t.type_label, "context": t.context or ""})
    with open(out / "taxonomy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["child", "parent"])
        w.writeheader()
        for e in dataset.taxonomy:
            w.writerow({"child": e.child, "parent": e.parent})
    with open(out / "relations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["head", "relation", "tail"])
        w.writeheader()
        for r in dataset.relations:
            w.writerow({"head": r.head, "relation": r.relation, "tail": r.tail})
    # also dump corpus
    corpus = dataset.metadata.get("corpus", [])
    if corpus:
        (out / "corpus.txt").write_text("\n".join(corpus) + "\n", encoding="utf-8")
