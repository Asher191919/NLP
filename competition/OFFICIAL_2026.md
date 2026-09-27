# LLMs4OL 2026 — Official Challenge Brief

**Compiled:** 2026-09-20  
**Challenge:** 3rd LLMs4OL Challenge @ ISWC 2026, Bari, Italy (25–29 October 2026)  
**Website:** https://sites.google.com/view/llms4ol2026  
**Repo:** https://github.com/sciknoworg/LLMs4OL-Challenge  
**Local clone:** `D:\NLP\data\LLMs4OL-Challenge\`  
**Contact:** llms4ol.challenge [at] gmail.com  
**Google Group:** https://groups.google.com/d/forum/llms4ol-challenge  
**Organizers:** Hamed Babaei Giglou, Jennifer D'Souza, Sören Auer (TIB)  
**Funding:** NFDI4DataScience (DFG 460234259), TIB

---

## 1. Task definitions and exact dataset fields

### Common 2026 shift
Unlike 2024 (Term Typing / Taxonomy / NonTax RE) and 2025 (Text2Onto / Term Typing / Taxonomy / NonTax RE), **2026 has three primary tasks** that package OL into realistic scenarios:

| Task | Name | One-line definition |
|------|------|---------------------|
| **A** | Flagship | Given raw text, construct a primitive ontology (terms, types, taxonomy, relations) |
| **B** | Reuse | Given a partial ontology + text, add new concepts and relations |
| **C** | Taxonomy | Given source-domain ontologies, induce hierarchy for an **unseen** domain |
| **D** | Ontologizer (optional, unevaluated) | Add ≥10 well-documented ontologies to OntoLearner via PR |

Two challenge phases:
1. **Solution Submission** — system outputs, scored with standardized metrics.
2. **System Integration** — integrate system into OntoLearner (mandatory for phase-1 participants).

---

### Task A — Flagship (End-to-End Ontology Learning)

**Definition:** Given raw text, construct a primitive ontology including terms, types, taxonomy, and relations.

**Must produce from text:** types, terms, term typings (`instance-of`), taxonomy (`is-a`), non-taxonomic relations, coherent graph.

**Local data:**
| File | Path | Samples |
|------|------|---------|
| Train | `data\LLMs4OL-Challenge\2026\TaskA-Flagship\train_task_a.json` | **4303** |
| Test  | `data\LLMs4OL-Challenge\2026\TaskA-Flagship\test_task_a.json`  | **2018** |

**Train fields (exact):**
```
id: string
context: string                    # "Title: ...\n\nContent:\n..."
primitive-ontology-triples: list[[subject, predicate, object], ...]
```

**Test fields (as published after release):**
```
id: string
context: string
terms: list[string]                # often []
types: list[string]
primitive-ontology-triples: list[[s,p,o], ...]   # gold present in public file
```

**Submission expectation (per official site):** predict `primitive-ontology-triples` for each test `id` given only `id` + `context` (terms/types may be empty or auxiliary).

**Triple predicates observed (open long-tail, not closed set):**  
`is-a`, `instance-of`, `is defined by`, `equivalent class`, `disjoint with`, `type`, `exact match`, `part_of`, `has part`, `database_cross_reference`, `see also`, chemical RO ids, etc.

**Contaminated ontologies — training on these is prohibited:**  
OBI, FOAF, CopyrightOnto, Metadata4Ing, PROCO, PTO, SWEET, SPDocument, MDSOnto, MatOnto, AgrO, TimelineOntology, MusicOntology, GTS, PeriodicTable, GND, QUDT, SchemaOrg, GeoNames, FoodOn, DOID, GoodRelations, BFO, ENVO, Conference, VIMMP, VIBSO, OM, DoCO, AUTO, Wine, DOLCE, CCO, DBpedia, MaterialInformation, LexInfo.

**Dataset URL:** https://github.com/sciknoworg/LLMs4OL-Challenge/tree/main/2026/TaskA-Flagship

---

### Task B — Reuse (Ontology Extension)

**Definition:** Given a partial ontology, add new concepts and relations extracted from text.

**Must produce:** term typings, taxonomy edges, non-taxonomic relations that **extend** the initial graph (not regenerate it).

**Local data:**
| File | Path | Samples |
|------|------|---------|
| Train | `...\2026\TaskB-Reuse\train_task_b.json` | **2774** |
| Test  | `...\2026\TaskB-Reuse\test_task_b.json`  | **1799** |

**Train fields (exact):**
```
id: string
context: string
terms: list[string]                              # often []
types: list[string]
initial-primitive-ontology-triples: list[triple] # input partial ontology
extended-primitive-ontology-triples: list[triple]# gold target extension
```

**Test fields:** same keys; system must predict `extended-primitive-ontology-triples`.

**Dataset URL:** https://github.com/sciknoworg/LLMs4OL-Challenge/tree/main/2026/TaskB-Reuse

Same contamination list as Task A.

---

### Task C — Taxonomy (Cross-Domain Taxonomy Discovery)

**Definition:** Given ontologies from source domains, induce a hierarchical taxonomy for an unseen domain.

**Input at test:** plain-text term/type lists per held-out ontology.  
**Output:** `is-a` edges, gold form `{"id","parent","child"}` ⇔ triple `(child, is-a, parent)`.

**Local gold / test-input files** (`...\2026\TaskC-Taxonomy\`):
| Domain file stem | Gold edges | Test input |
|------------------|------------|------------|
| Archeon | 237 | `Archeon_test_input.txt` |
| Biora | 156,430 | `Biora_test_input.txt` |
| Hylex | 657 | `Hylex_test_input.txt` |
| Phyra | 16,111 | `Phyra_test_input.txt` |

**Train data access:** **not** a static train JSON. Use **OntoLearner** (180+ ontologies, 22 domains):
```python
from ontolearner.ontology.events import Conference
ontology = Conference()
data = ontology.extract()
types = data.type_taxonomies.types
taxonomy = data.type_taxonomies.taxonomies  # [{ID, parent, child}, ...]
# submit form: (child, is-a, parent)
```

**Held-out test domains (training on these prohibited):**
- Material Science and Engineering (2–5 ontologies)
- Biology and Life Sciences (1–2)
- Scholarly Knowledge (2–5)
- Ecology and Environment (1–2)

**Train access instructions:** https://sites.google.com/view/llms4ol2026/taxonomy-task  
**HF collection:** https://huggingface.co/collections/SciKnowOrg/ontolearner-benchmarking

---

### Task D — Ontologizer (optional)

- Not part of scored evaluation.
- Extend OntoLearner ontology list (≥10 ontologies, single PR), with paper + 5-min video.
- Deadline tied to system integration end.
- Isolated paper required if also participating in A/B/C.

---

## 2. Metrics and formulas (especially Graph Similarity)

### Metric scripts — actual local paths
| Website claims | Actual cloned path |
|----------------|--------------------|
| `2026/graph_similarity_metric.py` | **`2026/metrics/graph_similarity.py`** |
| — | **`2026/metrics/taxonomy.py`** |

Both scripts load `SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)`.

### 2.1 Graph Similarity (Task A primary composite)

Official formula:

\[
\text{Graph Similarity} = \frac{\text{Edge F1} + \text{Neighborhood Similarity} + \text{Taxonomy Similarity}}{3}
\]

Each component ∈ \([0,1]\). Implementation detail from `graph_similarity.py`:

**Normalize** every triple token: `" ".join(str(x).lower().strip().split())`.

**Edge F1 (exact triple match)**  
Gold/pred as sets of `(s,p,o)`:
\[
P=\frac{|G\cap P|}{|P|},\quad R=\frac{|G\cap P|}{|G|},\quad F1=\frac{2PR}{P+R}
\]
Edge cases: both empty → 1.0; one empty → 0.0.

**Neighborhood Similarity (local structure)**  
For each node \(n\), build directed labeled neighborhood:
- outgoing: `(OUT, p, o)`
- incoming: `(IN, p, s)`

Then for every node in gold ∪ pred:
\[
J_n = \frac{|N_g(n)\cap N_p(n)|}{|N_g(n)\cup N_p(n)|}
\]
\[
\text{Neighborhood Sim} = \mathrm{mean}_n\, J_n
\]

**Taxonomy Similarity (hierarchical)**  
Build DiGraph of edges whose predicate ∈ `{"is-a"}` only. For each node:
\[
R_g(n)=\mathrm{ancestors}_g(n)\cup\mathrm{descendants}_g(n),\quad
R_p(n)=\mathrm{ancestors}_p(n)\cup\mathrm{descendants}_p(n)
\]
\[
J_n=\frac{|R_g\cap R_p|}{|R_g\cup R_p|},\quad
\text{Taxonomy Sim}=\mathrm{mean}_n\, J_n
\]

**Three matching modes** exposed by `evaluate(y_true, y_predict)`:
| Mode | Alignment | Threshold |
|------|-----------|-----------|
| `exact_match` | none (string identity after normalize) | — |
| `fuzzy_match` | Hungarian on rapidfuzz; triple score = mean of subject/predicate/object ratios | 0.90 |
| `semantic_match` | Hungarian on nomic-embed cosine; triple score = mean of s/p/o cosines | **0.90 in code** (site text also mentions 0.80 in places — use code) |

After alignment, edge_f1/neighborhood/taxonomy are computed on the **aligned** pred triples (fuzzy/semantic) or raw triples (exact).

### 2.2 Taxonomy metrics (Task C)

`metrics/taxonomy.py` — gold/pred as lists of `{parent, child}`:

**exact_match:** set F1 on lowercased `(parent, child)` pairs.

**fuzzy_match:** cost = 1 − mean(fuzz.ratio(parent), fuzz.ratio(child))/100; Hungarian; soft TP if sim ≥ 0.90; soft precision/recall/F1.

**semantic_match:** nomic-embeddings; pair sim = mean(parent cos, child cos); Hungarian; same soft TP at 0.90.

### 2.3 Standard subtask metrics
Official site also lists Precision / Recall / F1 for Term Typing, Taxonomy Discovery, and Non-Taxonomic RE as task-specific scores (Task B papers report these explicitly, e.g. pro-team: Typing F1 0.92, Taxonomy F1 0.854).

### 2.4 Known published system scores (reproduction anchors)
From pro-team (arXiv:2608.27101):
- Task A Semantic Graph Similarity: **0.7416**
- Task B Semantic Graph Similarity: **0.8692**
- Task B Term-Typing F1: **0.9200**
- Task B Taxonomy F1: **0.8540**
- Non-taxonomic RE: **0** (system extracted none)

---

## 3. Deadlines (all 2026 deadlines have passed as of 2026-09-20)

Source: https://sites.google.com/view/llms4ol2026/important-dates  
All times **23:59 AoE**.

| Milestone | Date(s) | Status |
|-----------|---------|--------|
| Website | 2026-03-30/31 | passed |
| Training datasets release | 2026-04-20 → **2026-04-26** | passed |
| Test datasets release | 2026-06-01 | passed |
| Begin system submissions | 2026-06-02 | passed |
| Intermediate submission | 2026-06-17 | passed |
| End system submissions | 2026-06-30 → **2026-07-07** | passed |
| Begin system integration | 2026-07-01 → **2026-07-08** | passed |
| End system integration | 2026-07-19 → **2026-07-26** | passed |
| Participant papers due | 2026-07-19 / 07-26 / **07-28** | passed |
| Notification | 2026-08-10 | passed |
| Camera-ready | 2026-08-31 | passed |
| ISWC 2026 challenge session | **2026-10-25 or 26** | upcoming |
| Ontologizer task deadline | end of system integration (site note also says June 30 — treat integration end as authoritative for Task D PRs) | passed |

**Conclusion:** LLMs4OL 2026 competition window is **closed**. Remaining 2026 activity = attend ISWC Bari workshop, consume accepted papers/proceedings, prepare **LLMs4OL 2027** and paper reproduction.

---

## 4. Accepted papers (LLMs4OL 2026)

Source: https://sites.google.com/view/llms4ol2026/accepted-papers

### Overview paper
- Hamed Babaei Giglou, Sören Auer, Jennifer D'Souza. *LLMs4OL 2026 Overview: The 3rd Large Language Models for Ontology Learning Challenge*.

### Long papers
1. **LaboroAI** — Xinyi Zhao, Jiachen Xu. *LaboroAI at LLMs4OL 2026 Tasks Flagship and Reuse: Less is More for LLM-based Ontology Learning.*
2. **ELLMO** — Ryn Gray, Ryan Roche, Jaimie Murdock, Douglas C. Crowder, Portia Cooper, Ena Nayak, Brendan Donohoe. *ELLMO at LLMs4OL 2026 Tasks Flagship, Reuse, and Taxonomy: Ontology Learning Across Contexts.*
3. **LettRAGraph** — Pasquale Lisena, Julien Plu, Oscar Moreno Escobar, Edouard Trouillez, Raphael Troncy. *LettRAGraph at LLMs4OL 2026 Task Flagship: Prompt-Guided Ontology Extraction with Ensembling.*
4. **pro-team** — Shivam Mishra, Dhannu Ram Meena, Muneendra Ojha, Krishna Pratap Singh, Kuldeep Singh. *pro-team at LLMs4OL 2026 Tasks Flagship and Reuse: Retrieval-Augmented Generation and Vocabulary-Constrained Filtering for Ontology Learning.* → arXiv:2608.27101 (local PDF + notes)
5. **Genuineknowledge-KEOL** — Lvting Liu, Yubin Luo, Ji Qi, Chun Zheng, Chengli Zhong, Yi Zhou. *Genuineknowledge-KEOL at LLMs4OL 2026 Tasks Flagship, Reuse, and Taxonomy: Hybrid Training-Free Pipelines for Ontology and Taxonomy Discovery.*
6. **VUL337-OL** — Chuhan Deng, Yuqi Fan, Can Zeng, Chao Zhang. *VUL337-OL at LLMs4OL 2026 Tasks Flagship and Reuse: Task-Native Constraints for Ontology Construction and Extension.*
7. **Volt-IQ** — Erban Ku. *Volt-IQ at LLMs4OL 2026: Constrained Graph Synthesis and Domain-Calibrated Ontology Learning.*
8. **semantic-swingers** — Matias Vizcaino, Jagdsh Chand, Andy Symonds. *semantic-swingers at LLMs4OL 2026 Tasks Flagship, Reuse, and Taxonomy: Retrieval, Fine-Tuning, and the Seen-Unseen Gap in LLM Ontology Learning.*

### Short papers
1. **LIST-CortAIx** — Sondes Bannour Souihi, Sandra Kara. *LIST-CortAIx at LLMs4OL 2026 Task Flagship: Stage-wise Design and Evaluation of a Modular LLM Pipeline for Ontology Construction.*
2. **XuanJi** — Jiaqi Jin. *XuanJi at LLMs4OL 2026 Task Flagship: A Partial Graph-Oriented System for Taxonomy and Typing.*
3. **HybridAI-Squared** — Upal Bhattacharya, Maaike de Boer, Sergey Sosnovsky. *HybridAI-Squared at LLMs4OL 2026 Task Reuse: Divide and Conquer does not help LLMs perform Ontology Learning.*
4. **SEMA** — Miquel Canal Esteve, Yoan Gutiérrez, José Abreu Salas, Fernando Llopis. *SEMA at LLMs4OL 2026 Task C: Continual Pretraining on Serialized Ontologies for Cross-Domain Taxonomy Discovery.*
5. **BigDipperAI** — Guojian Zhao, Yuxuan Tang, Zhao Li. *BigDipperAI at LLMs4OL 2026 Task Flagship: Structured Generation of Primitive Ontology Triples with Schema-Aligned Auxiliary Supervision.*

### Historical proceedings
- 2024 (1st): https://www.tib-op.org/ojs/index.php/ocp/issue/view/169
- 2025 (2nd): https://www.tib-op.org/ojs/index.php/ocp/issue/view/185
- 2025 overview DOI: https://doi.org/10.52825/ocp.v6i.2913

---

## 5. URLs (canonical index)

| Resource | URL |
|----------|-----|
| LLMs4OL 2026 site | https://sites.google.com/view/llms4ol2026 |
| Challenge overview | https://sites.google.com/view/llms4ol2026/challenge-overview |
| Flagship task | https://sites.google.com/view/llms4ol2026/flagship-task |
| Reuse task | https://sites.google.com/view/llms4ol2026/reuse-task |
| Taxonomy task | https://sites.google.com/view/llms4ol2026/taxonomy-task |
| Important dates | https://sites.google.com/view/llms4ol2026/important-dates |
| Accepted papers | https://sites.google.com/view/llms4ol2026/accepted-papers |
| Submission | https://sites.google.com/view/llms4ol2026/submission |
| Leaderboards | https://sites.google.com/view/llms4ol2026/leaderboards |
| Official data repo | https://github.com/sciknoworg/LLMs4OL-Challenge |
| 2026 folder | https://github.com/sciknoworg/LLMs4OL-Challenge/tree/main/2026 |
| Graph similarity script (site path) | https://github.com/sciknoworg/LLMs4OL-Challenge/blob/main/2026/graph_similarity_metric.py |
| Graph similarity script (actual) | https://github.com/sciknoworg/LLMs4OL-Challenge/blob/main/2026/metrics/graph_similarity.py |
| OntoLearner docs | https://ontolearner.readthedocs.io/ |
| OntoLearner GitHub | https://github.com/sciknoworg/OntoLearner |
| OntoLearner HF datasets | https://huggingface.co/collections/SciKnowOrg/ontolearner-benchmarking-6823bcd051300c210b7ef68a |
| OntoLearner metrics dashboard | https://huggingface.co/spaces/SciKnowOrg/OntoLearner-Benchmark-Metrics |
| ISWC 2026 | https://iswc2026.semanticweb.org/ |
| Series hub | https://sites.google.com/view/llms4ol |
| 2024 site | https://sites.google.com/view/llms4ol2024 |
| 2025 site | https://sites.google.com/view/llms4ol2025 |
| Google Group | https://groups.google.com/d/forum/llms4ol-challenge |
| Hamed baseline | https://github.com/HamedBabaei/LLMs4OL |
| Alexbek 2025 systems | https://github.com/BelyaevaAlex/LLMs4OL-Challenge-Alexbek |

### Key arXiv / DOI
| Paper | ID / DOI | Local |
|-------|----------|-------|
| LLMs4OL base (ISWC 2023) | arXiv:2307.16648 / 10.1007/978-3-031-47240-4_22 | `papers\refs\2307.16648.pdf` |
| 2024 overview | arXiv:2409.10146 / 10.52825/ocp.v4i.2473 | `papers\refs\2409.10146.pdf` |
| 2025 overview | 10.52825/ocp.v6i.2913 | (proceedings link) |
| Alexbek 2025 | arXiv:2508.19428 | `papers\refs\2508.19428.pdf` |
| pro-team 2026 | arXiv:2608.27101 | `papers\refs\2608.27101.pdf` |
| OntoLearner lib | arXiv:2607.01977 | `papers\refs\2607.01977.pdf` |

---

## 6. Local asset map (this machine)

```
D:\NLP\
├── competition\
│   ├── README.md              # prior OAEI vs LLMs4OL notes (mojibake encoding on disk)
│   └── OFFICIAL_2026.md       # this file
├── data\
│   ├── LLMs4OL-Challenge\     # official repo (2024+2025+2026 data & metrics)
│   │   └── 2026\
│   │       ├── TaskA-Flagship\{train,test}_task_a.json
│   │       ├── TaskB-Reuse\{train,test}_task_b.json
│   │       ├── TaskC-Taxonomy\{Archeon,Biora,Hylex,Phyra}_{gold.json,test_input.txt}
│   │       └── metrics\{graph_similarity.py,taxonomy.py}
│   ├── LLMs4OL-baseline\      # HamedBabaei/LLMs4OL (pre-cloned)
│   ├── OntoLearner\           # sciknoworg/OntoLearner clone
│   ├── alexbek\               # BelyaevaAlex 2025 systems
│   ├── ontolearner_pkg\       # pip --target install (if completed)
│   └── samples\               # schema samples + schema_notes.md
└── papers\refs\               # PDFs + citation markdown
```

---

## 7. Implications for 2027 and paper reproduction

### 7.1 Competition posture
- **2026 is closed.** Target **LLMs4OL 2027** (likely co-located with ISWC 2027) and/or the 2026 overview/proceedings as related work.
- OAEI 2026 execution deadline (2026-09-05) also passed; OAEI remains **backup** for 2027, LLMs4OL is **primary** (consistent with existing `competition/README.md` strategy).
- 2027 will almost certainly **require OntoLearner integration** in a second phase; design systems as `fit → predict → evaluate` learners now, not one-off scripts.

### 7.2 Metric-first engineering
- Implement and unit-test **Graph Similarity** early:
  \((\mathrm{EdgeF1}+\mathrm{NeighSim}+\mathrm{TaxSim})/3\)
  with the three modes (exact / fuzzy@0.90 / semantic@0.90, nomic-embed-text-v1.5).
- Leaderboards emphasize **Semantic Graph Similarity** — exact-string F1 alone underestimates near-miss surface forms and overestimates contamination-driven exact recall.
- Always report all three modes; papers that only report micro-F1 are not comparable to 2026 systems.

### 7.3 Data realities for reproduction
- Public 2026 test files **include gold triples/types**, so offline metric reproduction is possible without CodaLab.
- Task A train: 4303; Task B train: 2774; test: 2018 / 1799.
- Predicate distribution is **heavily `is-a` / `instance-of` skewed**; non-taxonomic predicates are rare → systems that ignore them still score well on graph similarity unless taxonomy/neighborhood terms compensate. This is an evaluation blind spot to discuss in a paper.
- Task C has no static train file — reproduction **requires OntoLearner + HF downloads**. Biora gold alone is 156k edges (large class imbalance vs Archeon 237).
- Contamination list is long and includes popular OL benchmarks; any reproduction using WordNet/UMLS/GeoNames/FoodOn/DOID/SWEET/MatOnto/PROCO/SchemaOrg training data violates challenge policy and inflates scores.

### 7.4 Method lessons from accepted systems
| Pattern | Evidence | 2027 action |
|---------|----------|-------------|
| Offline RAG + mid-size instruct LLM | pro-team Qwen2.5-14B + MiniLM retrieval | Strong, cheap baseline |
| Vocabulary-constrained filtering | pro-team Task B 0.8692 GS | Adopt for Reuse-style tasks |
| “Less is more” LLM pipelines | LaboroAI long paper | Ablate prompt/demo budget |
| Hybrid training-free pipelines | Genuineknowledge-KEOL | Useful when no finetune budget |
| Continual pretraining on serialized ontologies | SEMA (Task C short) | Direct 2027 Task C approach |
| Seen–unseen gap is real | semantic-swingers title/theme | Report seen vs blind splits separately |
| Divide-and-conquer does **not** help (Task B) | HybridAI-Squared short paper | Don’t assume modular RE/typing stacks win |
| Non-taxonomic RE ≈ 0 for RAG systems | pro-team | Explicitly open relation vocab or train RE head |

### 7.5 Contamination-aware research angle (this project’s thesis)
- LLMs4OL datasets are public ontology corpora (FoodOn, GO, schema.org, WordNet, …) — classic pretraining leakage surface.
- Graph Similarity’s exact Edge F1 can be **inflated by memorized is-a triples**; semantic mode may also reward memorized paraphrases.
- Recommended paper experiments:
  1. Membership-style probes on held-out vs train triples.
  2. Clean/polluted split reports (exclude banned ontology sources from any auxiliary retrieval index).
  3. Surface-form stress tests (synonym substitution, definition paraphrase) — graph similarity neighbor/taxonomy terms should degrade more than exact F1 if the model truly induces structure vs recalls edges.
  4. Compare OntoLearner-integrated learners vs raw LLM prompts under the same metric script.

### 7.6 Practical next steps (ordered)
1. Freeze local metric harness around `2026/metrics/*.py` + samples in `data\samples\`.
2. Install/complete OntoLearner env (`data\OntoLearner` + `data\ontolearner_pkg`); smoke-test `Conference().extract()`.
3. Reproduce one published number (pro-team Task B graph similarity) or a Hamed/Alexbek 2025 number as sanity check.
4. Build contamination probes on banned ontology list vs allowed training domains.
5. Draft 2027 system as OntoLearner learner module + graph-similarity-aware decoding (constraint: maximize neighborhood/taxonomy terms, not only triple precision).
6. Track ISWC 2026 Bari challenge session (Oct 25–26) and 2026 overview paper for official leaderboard tables.

---

## 8. Policy notes (from official site)

- Do **not** recreate train datasets from ontologies organizers held out for test.
- Extra WWW context is allowed **if documented** in paper/README.
- Papers should be research papers, not mere system reports.
- Peer review: each participant reviews ≥1 other team paper.
- Code must be shared on GitHub under an appropriate license.
- Phase 2 (OntoLearner integration) mandatory except Ontologizer-only participants.
- Scores may be withheld if submissions are incomplete/erroneous/deceptive.

---

*End of brief. Cross-check live site before citing any date as “final” for 2027 planning.*
