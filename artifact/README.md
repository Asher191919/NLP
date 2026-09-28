# OntoBridg Anonymous Artifact

This package supports offline verification of the manuscript *OntoBridg: Fail-Closed Evaluation of Open-Relation Ontology Learning*.

## Scope

The artifact contains:

- the anonymous manuscript, supplement, bibliography, and submission checklist;
- source code and a data-free portable unit-test subset;
- source-linked plotting code and publication figures;
- aggregate projections of the frozen result artifacts used by the manuscript checker;
- source-artifact SHA-256 values for provenance; and
- selection manifests that declare no protected gold or source text where applicable.

It intentionally excludes API credentials, raw generation caches, checkpoints, third-party datasets, source contexts, protected gold triples, and author metadata. Result JSON files in this package are publication projections, not byte-identical copies of the private research artifacts. Their source hashes are recorded in `SOURCE_ARTIFACTS.sha256`.

## Reproduction

Python 3.11 or newer is recommended.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_reproduction.py
```

The reproduction command performs an anonymity scan, verifies the package manifest, runs 88 data-free portable unit tests, checks manuscript tables and numeric claims against the aggregate evidence projections, regenerates both main figures, and verifies that the regenerated files match the packaged hashes.

The source repository's complete suite contains 125 tests. Fifteen tests require third-party challenge data or private experiment packets, and two data-bound modules contain another 22 tests. Those 37 tests are not executed by this public package because their required inputs are intentionally excluded. `run_portable_tests.py` fixes the 88-test public subset by test ID and fails if that selection drifts.

No command in this workflow makes an external model call. The original hosted-model generations cannot be reconstructed without provider access and the excluded private caches. Their prompts, system logic, attempt accounting, aggregate outputs, and source hashes remain auditable.

## Expected Result

The final line should be:

```text
Anonymous artifact reproduction verified
```

## Claim Boundary

This artifact supports a controlled synthetic positive result, official-training negative transfer results, and a fail-closed evaluation protocol. It does not support hidden-test, leaderboard, verified-contamination, naturalistic-gain, human-adjudication, or deployment claims.

## Licensing Note

No third-party dataset is redistributed. The upstream LLMs4OL repository is MIT-licensed, but some underlying ontology resources may impose additional terms. A project-level code and artifact license must be selected before public archival release.
