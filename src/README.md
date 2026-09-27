# OntoBridg

Hybrid **neuro-symbolic** ontology learning system targeting the LLMs4OL 2026/2027 task framing.

| Task | Name | Input | Output |
|------|------|-------|--------|
| A | Flagship | raw domain text | end-to-end ontology triples |
| B | Reuse | text + initial ontology + official terms/types | ontology extension triples |
| C | Taxonomy | cross-domain ontology inputs | `is-a` hierarchy edges |

Term extraction, term typing, taxonomy discovery, and non-taxonomic relation
extraction are internal pipeline stages. They are not relabeled as competition
Task A/B/C.

## Architecture

```
ontobridg/
  data/           loaders + WordNet (optional NLTK) + synthetic mock ontologies
  llm/            pluggable backends: mock (deterministic), OpenAI-compatible HTTP, local
  prompts/        instruction / cloze / few-shot templates for pipeline stages
  symbolic/       taxonomy transitivity, cycle break, domain/range + lexical filters
  pipeline/       extract → type → taxonomy → relations → symbolic repair
  eval/           micro/macro/per-class F1, taxonomy edge F1, hierarchical P/R, graph similarity
  contamination/  term holdout, paraphrase templates, novel synthetic ontologies
  cli/            run / eval / demo / contamination / export-synthetic
```

Neural path: LLM predicts term types, is-a edges, relations.  
Symbolic path: graph constraints repair cycles, fill limited transitive parents, filter illegal domain/range relations and lexical noise.

## Install

```powershell
cd D:\NLP\src
pip install -r requirements.txt
```

Python 3.10+ recommended. `requests` is only required for the OpenAI-compatible backend. Torch/transformers are **not** installed by default.

## Offline demo (no API key)

```powershell
& $env:MIMO_PYTHON D:\NLP\src\run_demo.py
# or
py -3 D:\NLP\src\run_demo.py
```

The mock LLM uses a **perturbed OntologyKnowledge** (dropped/flipped terms, noisy taxonomy/relations) so metrics are imperfect and symbolic repair is visible. Demo prints pipeline-component metrics, hierarchical graph similarity, contamination slices, and writes `D:\NLP\experiments\results\demo_metrics.json`. These synthetic metrics are not official competition results.

## CLI

```powershell
# Run mock pipeline on synthetic data
py -3 -m ontobridg.cli run --backend mock --output experiments\results\preds.json

# Evaluate predictions JSON against synthetic gold
py -3 -m ontobridg.cli eval --pred experiments\results\preds.json

# Contamination diagnostics
py -3 -m ontobridg.cli contamination

# Export synthetic dataset
py -3 -m ontobridg.cli export-synthetic --output experiments\results\synthetic.json
```

Competition folder layout for `--data`:

```
my_task/
  terms.csv       # term,type[,context]
  taxonomy.csv    # child,parent
  relations.csv   # head,relation,tail   (optional)
```

Or a single JSON ontology (see `export-synthetic` output schema).

## Plugging a real LLM API

OpenAI-compatible HTTP (OpenAI, DeepSeek, vLLM, LM Studio, Ollama `/v1`):

```powershell
$env:OPENAI_API_KEY = "sk-..."
# optional overrides
$env:ONTOBRIDG_BASE_URL = "https://api.openai.com/v1"
$env:ONTOBRIDG_MODEL = "gpt-4o-mini"

py -3 -m ontobridg.cli run --backend openai --model gpt-4o-mini --data path\to\dataset --output preds.json
```

Python API:

```python
from ontobridg.data.synthetic import make_synthetic_dataset, default_knowledge_from_dataset
from ontobridg.llm import build_llm
from ontobridg.pipeline import OntoBridgPipeline
from ontobridg.eval.metrics import evaluate_pipeline_result

dataset = make_synthetic_dataset()
# Path 1 (RAG / competition path): pass OntologyKnowledge
knowledge = default_knowledge_from_dataset(dataset, seed=13)
pipeline = OntoBridgPipeline(knowledge, use_symbolic_repair=True, seed=13)
result = pipeline.run_dataset(dataset)

# Path 2 (prompt / LLM path): pass LLM backend
llm = build_llm("mock", knowledge=knowledge)
pipeline2 = OntoBridgPipeline(llm, symbolic_repair=True, seed=13)
result2 = pipeline2.run(dataset)

report = evaluate_pipeline_result(
    dataset.term_map(),
    result.final_term_types(),
    dataset.taxonomy_pairs(),
    result.final_taxonomy(),
    gold_relations=dataset.relation_triples(),
    pred_relations=result.final_relations(),
)
print(report["task_a"]["f1"], report["task_b"]["f1"])
```

CLI（在 `D:\NLP\src` 下）:

```powershell
& $env:MIMO_PYTHON -m ontobridg --help
& $env:MIMO_PYTHON -m ontobridg.cli.main run --backend mock
```

Local OpenAI-compatible server:

```python
llm = build_llm("local", base_url="http://127.0.0.1:8000/v1", model="qwen2.5-7b-instruct")
```

Optional transformers path (requires torch): `ontobridg.llm.local.try_load_local_model(...)`.

## Prompt styles

- `instruction` — explicit schema + JSON response contract
- `cloze` — “`X` is a/an `Y`. ____” style (LLMs4OL-inspired)
- `few_shot` — short in-prompt demonstrations

Switch via `--prompt-style` or `OntoBridgPipeline(..., prompt_style="cloze")`.

## Symbolic layer

- `transitive_closure` / limited grandparent fill
- `detect_cycles` + edge-breaking repair
- `apply_type_range_filter` using `domain_ranges` + ancestor checks
- `lexical_relation_filter` for empty/self/unstable triples

Disable with `--no-symbolic` for ablation.

## Contamination utilities

- **Term holdout** — stratified split; score seen vs unseen terms
- **Paraphrase templates** — surface form shifts around heldout terms
- **Novel synthetic ontologies** — invented labels (`Zoranth`, `Vexul42`, …) to probe memorization

```powershell
py -3 -m ontobridg.cli contamination --holdout 0.3
```

Interpretation: large seen≫heldout or heldout≫paraphrase gaps suggest lexical memorization / surface sensitivity rather than ontology induction.

## WordNet (optional)

```python
from ontobridg.data.wordnet import try_load_wordnet_dataset
ds = try_load_wordnet_dataset(max_synsets=200)
```

Requires `nltk` + downloaded `wordnet` corpus. Returns `None` if unavailable (offline-safe).

## Results

Demo and CLI artifacts go to `D:\NLP\experiments\results\`.

Leakage-controlled official-train runners:

```powershell
# Prefix evaluation: Task A/B, closed vs open
& D:\NLP\src\run_blind_eval.ps1 -Task ab -Limit 30 -Workers 2 -VerboseRun

# Audit deterministic non-tax-density selection without making API calls
& D:\NLP\src\run_stratified_open_eval.ps1 -Task ab -SamplesPerStratum 5 -AuditOnly

# Legacy open vs guarded open; also derives guarded hybrid locally
& D:\NLP\src\run_stratified_open_eval.ps1 -Task ab -SamplesPerStratum 5 -Workers 2 -VerboseRun
```

`open_vocab_guarded` requires confidence and verbatim evidence for non-tax triples, rejects unsupported/generic/self-loop candidates, and enforces `-MaxNonTax` (default 8). The stratified runner samples five low (1), medium (2–4), and high (≥5) non-self non-tax gold cases per task using gold locally for selection only. Gold fields never enter model prompts. When both open systems are run, `open_vocab_guarded_hybrid` keeps legacy typing/taxonomy and substitutes the guarded relation channel without another API call.

### Guarded v2: canonicalization, implicit evidence, and calibration

`open_vocab_guarded_v2` adds three relation-only stages:

- predicate aliases and direction-aware canonicalization, plus conservative entity snapping to context and legal Task B input fields;
- `explicit` evidence with one verbatim span, or `cross_sentence` evidence with exactly two nearby verbatim spans and a stricter raw-confidence threshold;
- optional Beta-isotonic confidence calibration. A calibration artifact is applied only when it contains enough positive and negative candidates; degenerate artifacts fail closed with `deployable=false`.

The v2 cache has two independent identities. The prompt hash owns the raw model generation; the postprocess hash owns canonicalization, evidence gates, thresholds, and the calibrator. If only the postprocess hash changes, the runner reparses `raw_text` locally, updates the derived triples/diagnostics, and makes **zero external calls**. `--refresh-cache` is the explicit opt-in that forces a new generation.

```powershell
# Offline replay over existing guarded-v1 caches; makes zero API calls
py -3 D:\NLP\src\run_relation_v2_offline.py

# Rebuild the fail-closed pooled v5 grouped-CV audit; makes zero API calls
py -3 D:\NLP\src\run_gold_relation_grouped_cv.py

# Verify the immutable v5 confirmatory manifest and rebuild its score; zero API calls
py -3 D:\NLP\src\run_v5_confirmatory.py --audit-only --model dots3-note-prev
py -3 D:\NLP\src\run_v5_confirmatory.py --score-only --model dots3-note-prev

# Rebuild the locked error/bootstrap audit; zero API calls
py -3 D:\NLP\src\analyze_v5_confirmatory.py

# Verify the frozen semantic protocol and run the pinned secondary metric
py -3 D:\NLP\src\run_v5_semantic_secondary.py --freeze-only
$env:HF_HOME = 'D:\AI\hf-cache'
$env:HF_ENDPOINT = 'https://huggingface.co'
py -3 D:\NLP\src\run_v5_semantic_secondary.py

# Rebuild the closed-book factual-probe analysis; zero API calls
py -3 D:\NLP\src\analyze_official_contamination_probe.py

# Replay the endpoint-alias stress from cache, then run paired bootstrap; zero API calls
py -3 D:\NLP\src\run_official_contamination_surface_stress.py --model dots3-note-prev
py -3 D:\NLP\src\analyze_official_contamination_surface_stress.py

# Verify the frozen relation-rich nonce holdout and completed 60-unit cache; zero API calls
py -3 D:\NLP\src\build_nonce_relation_holdout.py --audit-only
py -3 D:\NLP\src\run_nonce_relation_holdout.py --audit-only --model dots3-note-prev

# Replay all completed nonce units from cache and rebuild the result; zero API calls
py -3 D:\NLP\src\run_nonce_relation_holdout.py --model dots3-note-prev --workers 2

# Verify the frozen official-text transfer packet and completed 60-unit cache; zero API calls
py -3 D:\NLP\src\build_official_relation_transfer_holdout.py --audit-only
py -3 D:\NLP\src\run_official_relation_transfer.py --model dots3-note-prev

# Verify the frozen, unexecuted predicate-routing packet and its 54-unit plan; zero API calls
py -3 D:\NLP\src\build_official_predicate_routing_holdout.py --audit-only
py -3 D:\NLP\src\run_official_predicate_routing.py --model dots3-note-prev

# Audit the v2 payload scope; makes zero API calls
& D:\NLP\src\run_stratified_open_eval.ps1 `
  -Task ab -SamplesPerStratum 5 `
  -Systems 'open_vocab,open_vocab_guarded_v2' -AuditOnly

# Run only after reviewing the audit and authorizing new external calls
& D:\NLP\src\run_stratified_open_eval.ps1 `
  -Task ab -SamplesPerStratum 5 -Workers 2 `
  -Systems 'open_vocab,open_vocab_guarded_v2' -VerboseRun
```

The 39-sample independent v2 calibration artifact contains 144 deployment-eligible candidates but only four exact positives from three samples, so it remains non-deployable under the five-positive-sample gate. A separate v5 pool has 31 candidates and seven positives from six samples; its grouped OOF metrics improve, but the former validation slice was inspected and then reused for calibration. `run_gold_relation_grouped_cv.py` therefore preserves the numeric result while forcing scientific deployment closed. `run_v5_confirmatory.py` freezes and executes a new balanced 24-sample endpoint-visible slice. On that slice, frozen calibration improves candidate Brier/ECE but reduces exact TP from four to three and fails the five-positive-sample gate, so `scientific_deployable=false` and `confirmatory_eligible=false` remain final. V2 and v5 are different systems and their calibration candidates must not be mixed.

`analyze_v5_confirmatory.py` adds a deterministic per-predicate error audit, 10,000-replicate sample-cluster intervals, and estimation-only scale planning. `run_v5_semantic_secondary.py` locks the official threshold, source hash, outer model revision, nested remote-code revision, and input hashes before scoring. At threshold 0.90 the semantic metric changes 0/24 confirmatory samples for both systems. Exact remains primary and neither analysis can authorize retuning or deployment.

The frozen contamination pilot uses 15 Task A and 15 Task B samples. The closed-book probe distinguishes 107 official facts from 107 constructed controls (AUROC 0.9780), but this is factual probeability rather than verified pretraining membership. The paired endpoint-alias stress has mixed directions: Task B GS drops under aliasing, whereas Task A point estimates improve. Both artifacts force `confirmatory_eligible=false`, make no deployment change, and contain no evidence about non-tax contamination on this sparse slice.

`nonce_relation_holdout_v1` is the frozen low-exposure follow-up: 30 synthetic context-only samples contain 120 balanced non-tax positives and 60 negated hard controls over 300 unique opaque entities. The model-input projection contains only `id` and `context`; gold remains local. All 60 locked units completed in 67 recorded attempts. Exact NonTax P/R/F1 is 0.2222/0.3167/0.2612 for `open_vocab` and 0.8644/0.8500/0.8571 for uncalibrated `open_vocab_guarded_v2`; both produce 0/60 hard-negative false positives. Guarded-v2 canonical/paraphrased F1 is 1.0000/0.7119. The result remains `confirmatory_eligible=false` and `scientific_deployable=false`: this is a controlled synthetic diagnostic, not an official, naturalistic, or verified temporal benchmark.

`official_relation_transfer_v1` is the completed transfer check for that synthetic result. It contains 30 unused, gold-stratified Task A training samples with 89 endpoint-visible and 101 total non-self non-tax gold relations. Equivalent-class, disjoint-with, and heterogeneous open-predicate strata contribute 10 samples each; executed and audit-only historical selections are excluded, and exact sample, compact-context, and non-tax-gold overlap with prior evidence are all zero. All 60 raw-open/guarded-v2 units completed in 81 recorded attempts. Exact NonTax P/R/F1 is 0/0/0 for both systems: guarded-v2 reduces non-self non-tax predictions from 424 to 143 and raises official exact GS from 0.1946 to 0.2593, but recovers 0/101 gold relations. The completed generation and postprocess caches audit at 60/60 hits with zero external calls. `analyze_official_relation_transfer.py` performs a zero-call post-hoc audit: all 225 raw guarded candidates fall outside the exact gold predicate inventory, with zero raw or rejected exact TP; raw open also emits no exact gold predicate, while 53/424 outputs are misserialized core aliases. This is diagnostic evidence of predicate-inventory mismatch, not permission to retune the inspected slice. `run_official_relation_transfer.py` remains audit-only unless `--execute` is explicitly supplied. The packet is negative gold-stratified official-training evidence, not an unbiased, hidden-test, naturalistic, temporal, or deployment benchmark.

`official_predicate_routing_v1` (package ID `78d478f4f1d9d76ff979`) is the frozen independent follow-up and has **NOT YET BEEN EXECUTED**. Its 27 Task A samples are allocated 8 equivalent-class / 8 disjoint-with / 3 type / 8 domain, with 64 endpoint-visible target gold edges and 69 total target gold edges. Prior sample-index, compact-context, and exact non-tax-gold overlap are all zero. The locked comparison is `open_vocab_guarded_v3_relations` versus `open_vocab_guarded_v6_predicate_routed`; prompt identity is the only system difference, with a shared postprocessor and no calibrator or output allowlist. The plan contains 54 units, with freeze-time generation and postprocess cache hits both 0/54 and 54 estimated external calls. The primary metric is four-stratum macro exact target F1. The mechanism gate requires routed-v6 macro F1 to be strictly greater than general-v3 and routed-v6 exact TP in at least five independent samples. Only `equivalent class`, `disjoint with`, `type`, and `database_cross_reference` have independent empirical support in this packet; type capacity is only three samples, and the other four predicates frozen in the v6 route have no empirical support here. Same-slice repair or retuning is prohibited. The two commands above are audit-only; supplying `--execute` requires separate authorization.

## Design notes

- Mock LLM is deterministic given `seed` + `OntologyKnowledge` — CI-safe offline evaluation.
- Symbolic repair is a **post-processing** stage so neural ablations stay measurable.
- Prefer stdlib; avoid torch by default for competition baselines on limited machines.
