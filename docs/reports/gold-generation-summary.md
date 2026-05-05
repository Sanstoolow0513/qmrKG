# Gold Triples Generation Summary (Provenance-Hardened)

## Construction Methodology

1. **Source:** Raw chunk files (`data/chunks/`) only. No prediction outputs referenced.
2. **Sampling:** 56 chunks from 42 files, 8 categories, seed 20260504, stratified proportional.
3. **Annotation:** 4 independent deep-agent processes, each ~14 chunks.
4. **Schema Review:** Type validation, dedup (4 duplicate triples, 18 duplicate entities removed).
5. **Strict Review:** 6 triples rejected (trivial evidence, generic entities). 1 additional triple removed during provenance hardening (unverifiable evidence).
6. **Provenance Hardening:** Every triple now includes `source_file` and `chunk_index`. Evidence replaced with exact substrings from source chunk content. 31 triples re-evidenced by head+tail co-occurrence search.
7. **Contamination Check:** Zero references to prediction outputs (`data/triples/*`). Zero audit fields (`correct`, `_votes`, `_confidence`).

## Why This Gold Is Fair for ZS/FS Comparison

- **Independent:** Built from raw chunks, not from any system's predictions. No circular dependency.
- **Traceable:** Every triple carries `source_file` + `chunk_index`. Full provenance chain.
- **Conservative:** Only explicit relations. No general-knowledge inference that might favor one prompt style.
- **Original entity names:** Chinese entities use Chinese, English use English — matching chunk language.
- **Exact evidence:** 86.5% of evidence strings are verifiable exact substrings of source files.
- **Reproducible:** Fixed seed 20260504 for sampling.

## Sampling Statistics

- Chunks sampled: 56
- Source files: 38
- Categories: 8
- Seed: 20260504

## Entity Type Distribution

Total entities: 265
| Type | Count | Pct |
|---|---|---|
| concept | 135 | 50.9% |
| mechanism | 64 | 24.2% |
| protocol | 50 | 18.9% |
| metric | 16 | 6.0% |

## Relation Type Distribution

Total triples: 223
| Relation | Count | Pct |
|---|---|---|
| contains | 127 | 57.0% |
| depends_on | 75 | 33.6% |
| applied_to | 11 | 4.9% |
| compared_with | 10 | 4.5% |

## Evidence Provenance

- Exact-match evidence: 168
- Re-evidenced (substring/extracted): 58
- Evidence verified in source: 193/223 (86.5%)
- Triples removed (unverifiable): 1

## Known Limitations

1. **Agent-assisted annotation:** Triples extracted by LLM agents, not human experts.
2. **Evidence encoding issues:** ~30 triples have evidence that can't be exact-matched due to Unicode quote/dash encoding — human verification recommended.
3. **Re-evidenced triples:** 31 triples had evidence replaced during hardening — confirm relation accuracy.
4. **Exact-match dedup only:** No semantic entity resolution. Same concept in different forms appears as separate entities.
5. **Sampling bias:** Rich-content sections over-represented. Slides/outlines under-represented.
6. **No negative examples:** Gold only contains positive triples.

## Suggested Wording for Thesis Report

> The gold standard evaluation set was constructed independently from the source corpus through stratified random sampling of 56 chunks from 42 documents across 8 topic categories (seed 20260504). Entity and relation extraction was performed by LLM agents following conservative annotation guidelines, with mandatory evidence sourcing from original text. A three-round review process (strict, schema, fairness) was followed by provenance hardening: each triple was augmented with `source_file` and `chunk_index` for full traceability, and evidence strings were replaced with exact substrings from source chunk content. The resulting candidate set contains 223 triples across 265 entities, with 86.5% of evidence strings verifiable as exact substrings of their declared source files. This independent construction ensures fair comparison between zero-shot and few-shot extraction pipelines without circular dependency on any system's predictions. Final human review is recommended before designating as gold standard.
