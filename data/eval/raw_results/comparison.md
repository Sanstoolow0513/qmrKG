# QmrKG Raw Extraction Evaluation: ZS vs FS

## Inputs

- ZS raw dir: `data/triples/raw-zs-recheck`
- FS raw dir: `data/triples/raw-fs-recheck`
- Gold: `data/eval/gold_triples.json`
- Compared at: `2026-05-06T11:02:54.873558Z`

## Scope-level Comparison (P / R / F1)

| Scope | Metric | ZS | FS | Δ (FS−ZS) |
| --- | --- | ---: | ---: | ---: |
| Entity | Precision | 0.4000 | 0.3690 | -0.0310 |
| Entity | Recall | 0.2862 | 0.2727 | -0.0135 |
| Entity | F1 | 0.3337 | 0.3136 | -0.0200 |
| Triple | Precision | 0.1644 | 0.1335 | -0.0309 |
| Triple | Recall | 0.1200 | 0.0980 | -0.0220 |
| Triple | F1 | 0.1387 | 0.1130 | -0.0257 |
| Relation | Precision | 0.5500 | 0.5385 | -0.0115 |
| Relation | Recall | 0.8049 | 0.6829 | -0.1220 |
| Relation | F1 | 0.6535 | 0.6022 | -0.0513 |

## Attribute Accuracy

| Attribute | ZS Acc | FS Acc | Δ (FS−ZS) |
| --- | ---: | ---: | ---: |
| head_type | 0.8667 | 0.8493 | -0.0174 |
| tail_type | 0.7524 | 0.8904 | +0.1380 |
| relation | 0.8571 | 0.8630 | +0.0059 |

## Notes

- **Entity**: `(name, type)` strict match
- **Triple**: `(head, head_type, relation, tail, tail_type)` strict match
- **Relation**: `(head_type, relation, tail_type)` match (ignoring entity names)
- **Attribute accuracy**: proportion of gold-aligned predictions where a single attribute matches gold
- Raw triples are type-enriched via the chunk's entity list before matching
- FPs only counted from chunks with gold coverage; FNs = gold triples not found in pred
