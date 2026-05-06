# Gold-500 Generation Summary

**Date**: 2026-05-05
**Verified**: 2026-05-06
**Seed**: 20260505
**Target**: 500 reviewed triples for zs/fs fair comparison evaluation

## Generation Pipeline

```
data/chunks/ (274 files, ~16,438 chunks)
    │
    ▼ Sampling (seed=20260505)
80 files, 225 chunks
    │
    ▼ Parallel extraction (8 agents)
403 raw candidates
    │
    ▼ Merge with existing 203 reviewed = 564 combined
    │
    ▼ Dedup (42 removed) + Schema validation (0 errors)
    │
    ▼ Generic entity filter (8 removed) + Evidence verification (16 fixed)
    │
    ▼ Semantics review (180 flagged, 40 rejected)
    │
    ▼ Filter to exactly 500
    │
    ▼ Final 500 triples, 594 entities
```

## Output Files

| File | Description |
|------|-------------|
| `data/eval/gold_triples.json` | **Canonical** 500-triple gold used by `kgeval` |
| `data/eval/gold_triples.500_reviewed.json` | Copy of the verified 500-triple gold |
| `data/eval/_rejection_log.json` | Retained rejection examples from the earlier review pass |
| `docs/reports/gold-500-generation-summary.md` | This file |
| `docs/reports/gold-500-provenance-audit.md` | Detailed provenance audit |
| `docs/reports/zs-fs-evaluation-500.md` | ZS/FS sampled-scope evaluation report |

## Statistical Profile

### Relation Distribution

| Relation | Count | Percentage |
|----------|-------|------------|
| `contains` | 304 | 60.8% |
| `depends_on` | 127 | 25.4% |
| `applied_to` | 48 | 9.6% |
| `compared_with` | 21 | 4.2% |

### Entity Type Distribution (head + tail)

| Type | Count | Percentage |
|------|-------|------------|
| protocol | 342 | 34.2% |
| concept | 411 | 41.1% |
| mechanism | 198 | 19.8% |
| metric | 49 | 4.9% |

### Source Diversity

- **88 unique source files** across 186 accepted source chunk pairs
- Categories covered: Chinese courseware, English textbooks, RFC specs, Chinese reference books, assignments, exam prep

### Layer Coverage

All OSI layers covered:
- Physical layer (signal encoding, media)
- Data link layer (Ethernet, 802.11, VLAN, CSMA/CD, ARP)
- Network layer (IPv4, IPv6, BGP, OSPF, RIP, routing, NAT, CIDR)
- Transport layer (TCP, UDP, SCTP, QUIC, congestion control)
- Application layer (HTTP, DNS, SMTP, DHCP, SIP)
- Wireless & mobile (5G, LTE, 802.11ax, MIMO, OFDM)
- Security (TLS, cryptography, DKIM)

### Protocol Coverage

| Protocol | Coverage |
|----------|----------|
| TCP | ✅ Extensive (congestion control, flow control, connection) |
| UDP | ✅ Good |
| IP/IPv4 | ✅ Good |
| IPv6 | ✅ Present (RFCs + textbooks) |
| BGP | ✅ Present (AS-PATH, loop prevention) |
| OSPF | ✅ Present (Dijkstra algorithm) |
| RIP | ✅ Present (distance vector) |
| DNS | ✅ Present |
| HTTP | ✅ Present (persistent/non-persistent) |
| TLS | ✅ Present (RFC 5246) |
| QUIC | ✅ Present |
| SCTP | ✅ Present (RFC 4960) |
| 802.11 | ✅ Present (frame format, comparison) |
| Ethernet/802.3 | ✅ Present |
| DHCP | ✅ Present |
| SMTP | ✅ Present (email system) |
| SIP | ✅ Present |
| DKIM | ✅ Present |

## Verification Results

| Check | Result |
|-------|--------|
| Triple count | ✅ 500/500 |
| No duplicates | ✅ 0 found |
| Evidence exact match | ✅ 500/500 after 17 whitespace-collapsed evidence repairs |
| Source file exists | ✅ 500/500 |
| Chunk index valid | ✅ 500/500 |
| Entity derivation | ✅ 594 derived, all from triples |
| Type validation | ✅ All protocol/concept/mechanism/metric |
| Relation validation | ✅ All contains/depends_on/compared_with/applied_to |
| Prediction-path reference check | ✅ Gold files contain no `data/triples/merged`, `raw-zs`, or `raw-fs` references |

## Known Limitations

1. **compared_with underweight**: 21 triples (4.2%) — could use more protocol comparison pairs
2. **Metric entities sparse**: 49 mentions (4.9%) — more quantitative entities would improve coverage
3. **Current-file isolation only**: the retained gold files contain no prediction-output paths, but file content alone cannot prove historical generation-time access.
4. **Alias normalization not performed** — entities kept in original form per spec (e.g., TCP vs TCP协议)
