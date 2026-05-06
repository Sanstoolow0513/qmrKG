# ZS/FS Evaluation Against 500-Triple Gold

**Date**: 2026-05-06
**Gold**: `data/eval/gold_triples.json` (`data/eval/gold_triples.500_reviewed.json` copy)
**Scope**: raw predictions are filtered to the 186 source chunk pairs referenced by the gold file before merging.
**Mapping**: `data/markdown/<name>.md + chunk_index` in raw outputs maps to `data/chunks/<name>.json + chunk_index` in gold provenance.
**Matching**: strict entity identity and strict triple identity; no alias normalization, suffix stripping, or semantic matching.

## Gold Verification Summary

- 500 triples and 594 entities.
- 88 source files and 186 source chunk pairs.
- 0 duplicate triples, 0 duplicate entities, 0 invalid schema references.
- 500/500 evidence strings are exact substrings of `data/chunks/<source_file>[chunk_index].content` after repairing 17 whitespace-collapsed evidence strings.
- Current gold files contain no references to `data/triples/merged`, `data/triples/raw-zs`, or `data/triples/raw-fs`; this verifies current file content, not historical generation-time access.

## Mode Comparison

| Mode | Raw Files | Raw Triples | Merged Triples | Entity TP | Entity P | Entity R | Entity F1 | Triple TP | Triple P | Triple R | Triple F1 | Evidence Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `zs-nocheck` | 184 | 418 | 337 | 307 | 0.2695 | 0.5168 | 0.3543 | 64 | 0.1899 | 0.1280 | 0.1529 | 1.0000 |
| `zs-recheck` | 182 | 397 | 314 | 305 | 0.2745 | 0.5135 | 0.3578 | 56 | 0.1783 | 0.1120 | 0.1376 | 1.0000 |
| `fs-nocheck` | 171 | 325 | 290 | 267 | 0.2436 | 0.4495 | 0.3160 | 47 | 0.1621 | 0.0940 | 0.1190 | 1.0000 |
| `fs-recheck` | 176 | 392 | 304 | 269 | 0.2604 | 0.4529 | 0.3307 | 48 | 0.1579 | 0.0960 | 0.1194 | 1.0000 |

## Interpretation

- `zs-nocheck` has the best strict triple F1 on the sampled-scope benchmark.
- `zs-recheck` has the best strict entity F1 on the same benchmark.
- Few-shot recheck increases triple recall over few-shot without recheck, but its precision drops because it also keeps more non-gold triples in the sampled chunks.
- Zero-shot without recheck is strongest under strict triple identity in this run; this is a conservative exact-match result, not a semantic equivalence judgment.
- Because the gold file is a reviewed positive reference set rather than a fully exhaustive annotation of every valid relation in each selected chunk, precision should be read as conservative strict-overlap precision. Recall over the 500 accepted gold triples is the more stable comparison signal.

## Reproduction

The evaluation script filters raw files by the `(source_file, chunk_index)` pairs in `data/eval/gold_triples.json`, maps raw `data/markdown/*.md` names to gold `data/chunks/*.json` names, merges with embedding disabled, and calls `qmrkg.evaluation.evaluate_files()`.
