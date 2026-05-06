# Gold-500 Provenance Audit

**Date**: 2026-05-06
**Gold files**: `data/eval/gold_triples.json`, `data/eval/gold_triples.500_reviewed.json`
**Verification tool**: Custom Python verification script + `qmrkg.evaluation`

## Provenance Chain

Every triple traces back through:
1. `source_file` → `data/chunks/<filename>.json` (274 files in corpus)
2. `chunk_index` → array index into source file's chunk array
3. `evidence` → exact substring of `chunks[chunk_index].content`

`data/eval/gold_triples.json` is now the canonical 500-triple gold file for `kgeval`;
`data/eval/gold_triples.500_reviewed.json` is an identical copy kept for audit clarity.

## Evidence Verification Methodology

```python
# For each triple:
chunks = json.load(Path("data/chunks") / triple["source_file"])
content = chunks[triple["chunk_index"]]["content"]
assert triple["evidence"] in content  # exact substring match
```

## Results

| Metric | Count |
|--------|-------|
| Total triples | 500 |
| Evidence exact match | 500 |
| Evidence failures | 0 |
| Source files verified | 88 unique |
| Source chunk pairs verified | 186 unique |
| Chunk indices verified | All valid |
| Evidence strings repaired | 17 whitespace-collapsed strings replaced with exact source spans |

## Source File Distribution (top 20 by triple count)

| Source File | Triples | Category |
|-------------|---------|----------|
| TCPIP详解 卷1：协议（原书第2版） | 27 | Chinese-reference-books |
| 计算机网络：自顶向下方法（原书第8版） (Kurose) | 22 | Chinese-course-materials |
| 计算机网络：自顶向下方法（原书第8版） ([美]版) | 22 | Chinese-course-materials |
| rfc4960.txt.json | 16 | RFC/protocol-specs |
| WNB-ch01b.json | 14 | Other-misc |
| 计算机网络（第6版） 自顶向下方法 | 11 | Chinese-course-materials |
| 第7章 无线网络和移动网络.json | 10 | Chinese-course-materials |
| Various others (80 files) | 378 | Mixed |

## Prediction-Path Reference Audit

The retained gold files were scanned for prediction-output references. No references to
`data/triples/merged`, `data/triples/raw-zs`, `data/triples/raw-fs`, or
`merged_triples.json` were found in:

- `data/eval/gold_triples.json`
- `data/eval/gold_triples.500_reviewed.json`
- `data/eval/sample_manifest.json`
- `data/eval/_rejection_log.json`

This is a current-file audit. It verifies that the retained gold data does not encode
prediction paths, but it cannot independently prove the full historical file access of
the generation process.

The `meta` field in the output confirms:
```json
"source_root": "data/chunks"
"gold_status": "reviewed_candidate_accepted_for_zs_fs_eval"
```

## Evidence Quality Assessment

| Evidence Characteristic | Count | Notes |
|------------------------|-------|-------|
| Exact source substring | 500 | Required for accepted gold |
| Whitespace-collapsed repair | 17 | Replaced with exact source spans on 2026-05-06 |
| Source-backed but not exhaustively semantic-reviewed | Remaining set | Suitable for strict zs/fs comparison with stated limitations |

## Recommendation for Future Iterations

1. **compared_with expansion**: Target 30-40 comparison triples for better coverage.
2. **Metric entity enrichment**: Extract more RTT, throughput, bandwidth, latency relationships.
3. **Exhaustiveness annotation**: If true precision is needed, annotate every valid relation in the selected chunks, not only accepted positive triples.
4. **Chinese-English alignment**: Some entities appear in both languages (Ethernet/以太网); document as aliases rather than merging in the strict gold file.
