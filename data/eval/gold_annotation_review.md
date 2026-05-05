# Gold Annotation Review

## Sampling Summary

- **Sampling unit:** chunk
- **Seed:** 20260504
- **Total valid chunks in corpus:** 14240
- **Sampled chunks:** 56 from 42 files

### Category Distribution

- Chinese-course-materials: 22
- English-textbooks: 9
- RFC/protocol-specs: 7
- Chinese-reference-books: 6
- Assignments-solutions: 3
- Labs-exercises: 3
- Other-misc: 3
- Review-exam-prep: 3

## Provenance Hardening (2026-05-04)

After initial generation, a provenance hardening pass was applied:

1. **Added `source_file` and `chunk_index` to every triple** — enabling full traceability back to the original chunk.
2. **Replaced agent-summarized evidence with exact substrings** from source chunk content.
3. **31 triples re-evidenced** by searching for head+tail entity co-occurrence in source chunks.
4. **1 triple removed** — evidence could not be verified in any source chunk (IP地址 → 网络号).
5. **Evidence verification:** 193/223 (86.5%) of evidence strings are exact substrings of their declared source_file.

## Issues Requiring Human Review

### Evidence Encoding Issues (~30 triples)
Approximately 30 triples have evidence that could not be confirmed as exact substrings of the source file, likely due to Unicode encoding differences (smart quotes `“` vs `"`, em-dashes `—` vs `--`). These should be spot-checked manually.

### Re-Evidenced Triples (31 triples)
These triples had their original agent-written evidence replaced during provenance hardening. The new evidence was extracted by searching for head+tail co-occurrence in source chunks. Human review should confirm that the extracted evidence supports the claimed relation.

### Entity Name Normalization
Some entities have near-duplicate forms across different chunks (e.g., 拥塞控制 vs 拥塞控制算法, TCP vs TCP协议). These are kept in original form. Human reviewers should decide whether to merge or document as-is.

### Coverage Gaps
- `compared_with` relations are sparse (10 triples). Consider expanding.
- Metric-type entities (16) are underrepresented vs concepts (135).
- DNS, BGP, IPv6, VLAN, MPLS are under-represented due to sampling distribution.

### Known Entity Alias / Normalization Candidates

- `信息共享` ↔ `共享` (both concept)
- `IP 数据报` ↔ `数据报` (both concept)
- `拥塞控制` ↔ `拥塞控制算法` (both mechanism)
- `802.11` ↔ `802.11ax` (both protocol)
- `802.3` ↔ `IEEE 802.3` (both protocol)
- `三次握手` ↔ `握手` (both mechanism)
- `T/TCP` ↔ `TCP` (both protocol)
- `DHCP` ↔ `DHCPv6` (both protocol)
- `EstimatedRTT` ↔ `RTT` (both metric)
- `OFDM` ↔ `OFDMA` (both mechanism)
- `持续 HTTP` ↔ `非持续 HTTP` (both mechanism)
- `4G LTE` ↔ `LTE` (both protocol)
- `基于连接的服务` ↔ `连接` (both concept)
- `无连接服务` ↔ `连接` (both concept)