"""Optional WordNet loader — falls back gracefully when nltk/corpora unavailable."""
from __future__ import annotations

from ontobridg.data.schema import OntologyDataset


def try_load_wordnet_dataset(limit: int = 200) -> OntologyDataset | None:
    """Attempt to load a small WordNet-derived ontology dataset.

    Returns None if WordNet is unavailable (expected on offline/mock setups).
    """
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except Exception:
        return None
    try:
        wn.synsets("dog")
    except Exception:
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            wn.synsets("dog")
        except Exception:
            return None
    terms = []
    taxonomy = []
    seen = set()
    for pos in ("n",):
        for syn in list(wn.all_synsets(pos))[: limit * 2]:
            name = syn.name().split(".")[0]
            if name in seen:
                continue
            seen.add(name)
            lexname = syn.lexname() or "noun.Tops"
            terms.append(type("T", (), {"term": name, "type_label": lexname, "context": None})())
            for hyp in syn.hyponyms():
                hname = hyp.name().split(".")[0]
                if hname not in seen:
                    taxonomy.append(type("E", (), {"child": hname, "parent": name})())
            if len(terms) >= limit:
                break
        if len(terms) >= limit:
            break
    from ontobridg.data.schema import TermSample, TaxonomySample

    return OntologyDataset(
        name="wordnet_nouns",
        types=sorted({t.type_label for t in terms}),
        terms=[TermSample(term=t.term, type_label=t.type_label) for t in terms],
        taxonomy=[TaxonomySample(child=e.child, parent=e.parent) for e in taxonomy],
        relations=[],
    )
