# QmrKG Evaluation Report

## Evaluation Inputs

- Prediction file: `data/triples/merged/merged_triples.json`
- Gold file: `data/eval/gold_pure.json`
- Evaluated at: `2026-05-01T15:39:55.162130Z`
- Gold schema version: `1`
- Matching: strict entity and triple identity

## Summary

| Scope | Pred | Gold | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Entity | 2423 | 99 | 29 | 2394 | 70 | 0.0120 | 0.2929 | 0.0230 |
| Triple | 1125 | 54 | 0 | 1125 | 54 | 0.0000 | 0.0000 | 0.0000 |

## Evidence

Predicted evidence coverage: 1125/1125 (1.0000)
True-positive evidence coverage: 0/0 (0.0000)

## Error Samples

### False Positives

| Head | Head Type | Relation | Tail | Tail Type |
| --- | --- | --- | --- | --- |
| `/23` | `metric` | `applied_to` | `Lan D` | `concept` |
| `/24` | `metric` | `applied_to` | `Lan A` | `concept` |
| `/25` | `metric` | `applied_to` | `Lan B` | `concept` |
| `/25` | `metric` | `applied_to` | `Lan C` | `concept` |
| `1000BASE-T` | `protocol` | `applied_to` | `汇聚层` | `concept` |
| `1000BASE-T` | `protocol` | `depends_on` | `1000BASE-X` | `protocol` |
| `1000BASE-T` | `protocol` | `depends_on` | `4 个线对` | `concept` |
| `1000BASE-T` | `protocol` | `depends_on` | `5 类无屏蔽双绞线` | `concept` |
| `100BASET` | `concept` | `applied_to` | `光纤` | `concept` |
| `100BASET` | `concept` | `applied_to` | `双绞线` | `concept` |

### False Negatives

| Head | Head Type | Relation | Tail | Tail Type |
| --- | --- | --- | --- | --- |
| `5层因特网协议栈` | `concept` | `contains` | `运输层` | `concept` |
| `AP` | `concept` | `applied_to` | `RADIUS鉴别服务器` | `concept` |
| `A类地址` | `concept` | `compared_with` | `B类地址` | `concept` |
| `CSMA/CA` | `protocol` | `applied_to` | `ACK帧` | `concept` |
| `CSMA/CA` | `protocol` | `depends_on` | `二进制指数退避` | `mechanism` |
| `FORM` | `concept` | `contains` | `input` | `concept` |
| `GBN` | `protocol` | `applied_to` | `信道利用率` | `metric` |
| `GSM` | `protocol` | `contains` | `MSC` | `concept` |
| `HTTP 首部字段` | `concept` | `contains` | `请求首部字段` | `concept` |
| `HTTP头压缩` | `mechanism` | `applied_to` | `请求头` | `concept` |

## Notes

- Entity matches require exact `name` and `type` equality.
- Triple matches require exact `head`, `head_type`, `relation`, `tail`, and `tail_type` equality.
- Evidence coverage measures predicted triples with non-empty evidence fields.
