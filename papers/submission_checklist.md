# OntoBridg Submission Checklist

This checklist tracks manuscript packaging rather than new model development. The current target is a venue-neutral workshop manuscript that can be moved into the LLMs4OL 2027 or ISWC workshop template once the official call is available.

## Scientific Scope

- [x] Main claim is a controlled positive result plus independent negative transfer evidence.
- [x] Public training-data results are not described as hidden-test, leaderboard, temporal, or deployment results.
- [x] Exact false positives are distinguished from human-adjudicated semantic errors.
- [x] Weak labels are identified as dual-model weak labels, not human annotations.
- [x] Calibration, mechanism, confirmatory, and deployment gates remain closed.
- [x] Same-slice prompt repair and n>=100 expansion are excluded after the failed routing gate.

## Manuscript

- [x] Anonymous, venue-neutral English manuscript with no local absolute paths.
- [x] Contributions correspond to executed experiments.
- [x] Aspirational RAG, Task C, and multi-objective ablations removed from the evaluated method claim.
- [x] Main tables use official exact metrics and identify training-data scope.
- [x] Limitations cover one hosted model, sparse gold, synthetic data, contamination inference, licensing, and human-audit absence.
- [x] Supplement contains extended calibration, transfer, routing, oracle-pilot, and API-stability details.
- [x] BibTeX file contains verified metadata for all cited work.
- [x] Main result tables and headline numeric claims are checked against frozen JSON artifacts.
- [x] Internal adversarial review distinguishes gold-withheld prompting from pretraining contamination and local protocol freezing from public preregistration.
- [ ] Add author names, affiliations, ORCID identifiers, and corresponding-author email after anonymity requirements are known.
- [ ] Convert Markdown to the official CEUR/LNCS template after the venue is selected.
- [ ] Recheck page limit after template conversion.

## Figures and Tables

- [x] Figure 1 presents the method and fail-closed evidence chain.
- [x] Figure 2 presents nonce, official-transfer, and routing outcomes without changing scales or measurements.
- [x] Vector SVG/PDF and 300 dpi PNG exports retained with editable plotting source.
- [x] Weak-label reliability figure moved to the supplement.
- [ ] Recheck figure width and font size in the final venue template.
- [ ] Add alt text or accessibility descriptions if required by the venue.

## Artifact and Reproducibility

- [x] Frozen manifests, prompt hashes, result JSON, and attempt ledgers retained.
- [x] Generation and postprocessing caches have separate identities.
- [x] Zero-call audit commands documented.
- [x] API credentials, caches, checkpoints, and third-party datasets excluded from Git.
- [x] Full 170-test suite passes in the source workspace with retained third-party and private experiment inputs.
- [x] Verify the anonymous artifact's fixed 129-test, data-free portable subset in a clean virtual environment.
- [ ] Create a public archival release and DOI after repository URL/anonymity policy is decided.
- [ ] Add the final repository URL and commit/tag to the camera-ready artifact statement.
- [x] Verify that the anonymous artifact contains no user name, local path, or provider credential.

## Data and Ethics

- [x] Challenge repository license identified as MIT.
- [x] Upstream ontology licensing caveat retained, especially for UMLS-derived resources.
- [x] No personal data or human-subject claims introduced.
- [x] Model provenance and provider-side drift limitation disclosed.
- [ ] Perform a final third-party data redistribution review before public release.

## Venue Decision

- [ ] Verify the official LLMs4OL 2027 call, dates, page limit, anonymity policy, and artifact policy.
- [ ] If no suitable 2027 workshop call appears, evaluate ISWC workshop/negative-results fit before a full research-track submission.
- [ ] Do not submit the current evidence package as an ISWC full research paper without broader model or dataset replication.
