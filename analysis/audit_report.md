# Raw Triple Quality Audit Report

**Sample size:** 500 files out of 16438 total
**Triples sampled:** 2244
**Entities sampled:** 3028

## Category 1: 幻觉 / Evidence 不支撑

- **Triples checked:** 2244
- **Flagged (score < 0.8):** 994 (44.3%)
- **Average evidence score:** 0.777
- **Issue breakdown:**
  - `EVIDENCE_NOT_IN_SOURCE`: 906
  - `HEAD_NOT_IN_EVIDENCE`: 197
  - `NO_ENTITY_IN_EVIDENCE`: 140
  - `TAIL_NOT_IN_EVIDENCE`: 80

### Flagged Examples (Hallucination)

- **File:** `Chapter_4_V7.01_chunk_0010.json`
  - Triple: (IP) -[contains]-> (datagram format)
  - Evidence: IP: Internet Protocol - datagram format
  - Score: 0.70 | Issues: ['EVIDENCE_NOT_IN_SOURCE']

- **File:** `Chapter_4_V7.01_chunk_0010.json`
  - Triple: (IP) -[contains]-> (fragmentation)
  - Evidence: IP: Internet Protocol - fragmentation
  - Score: 0.70 | Issues: ['EVIDENCE_NOT_IN_SOURCE']

- **File:** `Chapter_4_V7.01_chunk_0010.json`
  - Triple: (IP) -[contains]-> (IPv4 addressing)
  - Evidence: IP: Internet Protocol - IPv4 addressing
  - Score: 0.70 | Issues: ['EVIDENCE_NOT_IN_SOURCE']

- **File:** `Chapter_4_V7.01_chunk_0010.json`
  - Triple: (IP) -[contains]-> (IPv6)
  - Evidence: IP: Internet Protocol - IPv6
  - Score: 0.70 | Issues: ['EVIDENCE_NOT_IN_SOURCE']

- **File:** `Chapter_4_V7.01_chunk_0010.json`
  - Triple: (IP) -[contains]-> (Network address translation)
  - Evidence: IP: Internet Protocol - network address translation
  - Score: 0.50 | Issues: ['TAIL_NOT_IN_EVIDENCE', 'EVIDENCE_NOT_IN_SOURCE']

- **File:** `Chapter_4_V7.01_chunk_0010.json`
  - Triple: (IPv4) -[compared_with]-> (IPv6)
  - Evidence: IP: Internet Protocol - IPv4 addressing, IPv6
  - Score: 0.70 | Issues: ['EVIDENCE_NOT_IN_SOURCE']

- **File:** `Chapter_4_V7.01_chunk_0010.json`
  - Triple: (Network Layer) -[contains]-> (Data plane)
  - Evidence: 4.1 Overview of Network layer - data plane
  - Score: 0.30 | Issues: ['NO_ENTITY_IN_EVIDENCE', 'EVIDENCE_NOT_IN_SOURCE']

- **File:** `Chapter_4_V7.01_chunk_0010.json`
  - Triple: (Network Layer) -[contains]-> (Control plane)
  - Evidence: 4.1 Overview of Network layer - control plane
  - Score: 0.30 | Issues: ['NO_ENTITY_IN_EVIDENCE', 'EVIDENCE_NOT_IN_SOURCE']

- **File:** `rfc9112_chunk_0027.json`
  - Triple: (HTTP) -[contains]-> (HTTP/1.1)
  - Evidence: RFC 9112 HTTP/1.1
  - Score: 0.70 | Issues: ['EVIDENCE_NOT_IN_SOURCE']

- **File:** `rfc9112_chunk_0027.json`
  - Triple: (ALPN) -[applied_to]-> (HTTP/1.1)
  - Evidence: IANA has updated the 'TLS Application-Layer Protocol Negotiation (ALPN) Protocol IDs' registry with the registration for HTTP/1.1
  - Score: 0.70 | Issues: ['EVIDENCE_NOT_IN_SOURCE']


## Category 2: 类型选错

- **Entities checked:** 3028
- **Flagged (suspicious type):** 389 (12.8%)
- **Type distribution in sample:**
  - `concept`: 1738
  - `protocol`: 770
  - `mechanism`: 310
  - `metric`: 210
- **Issue breakdown:**
  - `LIKELY_MECHANISM_GOT_CONCEPT`: 182
  - `LIKELY_PROTOCOL_GOT_CONCEPT`: 171
  - `LIKELY_PROTOCOL_GOT_MECHANISM`: 20
  - `LIKELY_METRIC_GOT_CONCEPT`: 15
  - `LIKELY_MECHANISM_GOT_PROTOCOL`: 7
  - `LIKELY_MECHANISM_GOT_METRIC`: 4
  - `LIKELY_PROTOCOL_GOT_METRIC`: 1

### Flagged Examples (Type Mismatch)

- `IP Precedence`: actual=`concept` → suggested=`protocol` (['LIKELY_PROTOCOL_GOT_CONCEPT'])
- `路由算法`: actual=`concept` → suggested=`mechanism` (['LIKELY_MECHANISM_GOT_CONCEPT'])
- `静态路由算法`: actual=`concept` → suggested=`mechanism` (['LIKELY_MECHANISM_GOT_CONCEPT'])
- `动态路由算法`: actual=`concept` → suggested=`mechanism` (['LIKELY_MECHANISM_GOT_CONCEPT'])
- `距离-向量路由算法`: actual=`concept` → suggested=`mechanism` (['LIKELY_MECHANISM_GOT_CONCEPT'])
- `链路状态路由算法`: actual=`concept` → suggested=`mechanism` (['LIKELY_MECHANISM_GOT_CONCEPT'])
- `路由表`: actual=`concept` → suggested=`mechanism` (['LIKELY_MECHANISM_GOT_CONCEPT'])
- `选择重传协议`: actual=`protocol` → suggested=`mechanism` (['LIKELY_MECHANISM_GOT_PROTOCOL'])
- `TCP/IP 模型`: actual=`concept` → suggested=`protocol` (['LIKELY_PROTOCOL_GOT_CONCEPT'])
- `CSMA/CA`: actual=`mechanism` → suggested=`protocol` (['LIKELY_PROTOCOL_GOT_MECHANISM'])


## Category 3: 明显遗漏

- **Chunks checked:** 500
- **Flagged (suspicious omission):** 90 (18.0%)
- **Issue breakdown:**
  - `ZERO_EXTRACTION_DESPITE_CONTENT`: 74
  - `LOW_EXTRACTION_RATIO`: 51
  - `PROTOCOLS_MENTIONED_BUT_NOT_EXTRACTED`: 14

### Flagged Examples (Omission)

- **File:** `Chapter_1_V7.01_chunk_0009.json`
  - Titles: ['Chapter 1: roadmap']
  - Content length: 323 | Entities: 0 | Triples: 0
  - Issues: ['ZERO_EXTRACTION_DESPITE_CONTENT']
  - Content preview: # Chapter 1: roadmap

1.1 what is the Internet?
1.2 network edge
　　- end systems, access networks, links
1.3 network core
　　- packet switching, circui

- **File:** `计算机网络第8版课件-第6章-应用层_chunk_0206.json`
  - Titles: ['Get-request 报文 ASN.1 定义']
  - Content length: 436 | Entities: 0 | Triples: 0
  - Issues: ['ZERO_EXTRACTION_DESPITE_CONTENT']
  - Content preview: # Get-request 报文 ASN.1 定义

Get-request-PDU :: ● [0]
--[0] 表示上下文类，编号为 0

IMPLICIT SEQUENCE {
-- 类型是 SEQUENCE

request-id integer32,
-- 变量 request-id 的类

- **File:** `计算机网络第8版课件-第7章 网络安全_chunk_0069.json`
  - Titles: ['安全关联的特点']
  - Content length: 260 | Entities: 0 | Triples: 0
  - Issues: ['ZERO_EXTRACTION_DESPITE_CONTENT', 'PROTOCOLS_MENTIONED_BUT_NOT_EXTRACTED']
  - Content preview: # 安全关联的特点

- 安全关联是从源点到终点的单向连接，它能够提供安全服务。
- 在安全关联 SA 上传送的就是 IP 安全数据报。
- 如要进行双向安全通信，则两个方向都需要建立安全关联。
- 若 n 个员工进行双向安全通信，一共需要创建（2 + 2n）条安全关联 SA。

```icon: 

- **File:** `Chapter_1_V7.01_chunk_0000.json`
  - Titles: ['Chapter 1 Introduction']
  - Content length: 933 | Entities: 0 | Triples: 0
  - Issues: ['ZERO_EXTRACTION_DESPITE_CONTENT', 'LOW_EXTRACTION_RATIO']
  - Content preview: # Chapter 1 Introduction

A note on the use of these Powerpoint slides:
We're making these slides freely available to all (faculty, students, readers)

- **File:** `Chapter6-Application Layer_chunk_0040.json`
  - Titles: ['SMTP交互示例']
  - Content length: 521 | Entities: 4 | Triples: 0
  - Issues: ['LOW_EXTRACTION_RATIO']
  - Content preview: # SMTP交互示例

6.1 网络应用体系结构
6.2 网络应用通信原理
6.3 域名解析系统(DNS)
6.4 FTP应用
6.5 Email应用

S：220 hamburger.edu
C：HELO crepes.fr
S：250 Hello crepes.fr, pleased to me

- **File:** `Chapter_8_V7.0_chunk_0124.json`
  - Titles: ['Intrusion detection systems']
  - Content length: 322 | Entities: 0 | Triples: 0
  - Issues: ['ZERO_EXTRACTION_DESPITE_CONTENT', 'PROTOCOLS_MENTIONED_BUT_NOT_EXTRACTED']
  - Content preview: # Intrusion detection systems

multiple IDSs: different types of checking at different locations

```icon: firewall
```

```icon: internal network
```

- **File:** `Chapter1-Introduction_chunk_0092.json`
  - Titles: ['例题3']
  - Content length: 254 | Entities: 0 | Triples: 0
  - Issues: ['ZERO_EXTRACTION_DESPITE_CONTENT']
  - Content preview: # 例题3

1.1 计算机网络基本概念
1.2 计算机网络结构
1.3 数据交换
1.4 计算机网络性能指标

【例2】如下图所示网络，主机A通过路由器R1和R2连接主机B，三段链路带宽分别是100kbps、2Mbps和1Mbps。假设A以存储-转发的分组交换方式向B发送一个大文件。

请回答下列

- **File:** `王道计算机408考研-计算机组成原理网课PPT讲义-分章节带目录 (王道考研) (z-library.sk, 1lib.sk, z-lib.sk)_chunk_0006.json`
  - Titles: ['硬件的发展']
  - Content length: 544 | Entities: 0 | Triples: 0
  - Issues: ['ZERO_EXTRACTION_DESPITE_CONTENT', 'LOW_EXTRACTION_RATIO']
  - Content preview: # 硬件的发展

| 发展阶段 | 时间       | 逻辑元件         | 速度(次/秒) | 内存         | 外存           |
|----------|------------|------------------|-------------|----------

- **File:** `5G_chunk_0016.json`
  - Titles: ['趋势 2 ：智能终端引领移动互联的发展']
  - Content length: 544 | Entities: 0 | Triples: 0
  - Issues: ['ZERO_EXTRACTION_DESPITE_CONTENT', 'LOW_EXTRACTION_RATIO']
  - Content preview: # 趋势 2 ：智能终端引领移动互联的发展

- 智能终端推动应用服务快速创新，移动互联网以 6 个月为周期快速迭代
  -- 桌面互联网 /PC ：以 18 个月为周期进行硬件和软件升级
  -- 移动互联网 / 智能终端：产业周期趋近于 6 个月

应用服务快速创新

操作系统
iOS
andr

- **File:** `rfc2328.txt_chunk_0083.json`
  - Titles: ['16. Calculation of the routing table', '16.4.1. External path preferences']
  - Content length: 1099 | Entities: 0 | Triples: 0
  - Issues: ['ZERO_EXTRACTION_DESPITE_CONTENT', 'LOW_EXTRACTION_RATIO']
  - Content preview: ## 16.4.1. External path preferences

When multiple intra-AS paths are available to
ASBRs/forwarding addresses, the following rules indicate
which pat


## Category 4: 实体名不一致 / 噪声

- **Entities checked:** 3028
- **Flagged (name quality issue):** 467 (15.4%)
- **Intra-chunk conflicts (near-duplicates):** 380
- **Name length distribution:**
  - `1-5` chars: 1849
  - `6-10` chars: 700
  - `11-20` chars: 393
  - `21-30` chars: 86
  - `31+` chars: 0
- **Issue breakdown:**
  - `CONTAINS_BRACKETS_OR_DIGITS`: 258
  - `MIXED_LANG_NO_SUFFIX`: 224
  - `CONTAINS_PUNCTUATION`: 6

### Flagged Examples (Name Quality)

- `IPv4`: ['CONTAINS_BRACKETS_OR_DIGITS']
- `IPv6`: ['CONTAINS_BRACKETS_OR_DIGITS']
- `IP`: SUBSTRING_MATCH
- `IP`: SUBSTRING_MATCH
- `HTTP/1.1`: ['CONTAINS_BRACKETS_OR_DIGITS']
- `HTTP`: SUBSTRING_MATCH
- `compress`: SUBSTRING_MATCH
- `gzip`: SUBSTRING_MATCH
- `Type of Service (TOS)`: ['CONTAINS_BRACKETS_OR_DIGITS']
- `路由算法`: SUBSTRING_MATCH
- `路由算法`: SUBSTRING_MATCH
- `路由算法`: SUBSTRING_MATCH
- `TCP/IP 模型`: ['MIXED_LANG_NO_SUFFIX']
- `P2P 模型`: ['CONTAINS_BRACKETS_OR_DIGITS', 'MIXED_LANG_NO_SUFFIX']
- `rdt2.x`: ['CONTAINS_BRACKETS_OR_DIGITS']


## Relation Quality

- **Triples checked:** 2244
- **Relation distribution:**
  - `contains`: 820
  - `applied_to`: 653
  - `depends_on`: 462
  - `compared_with`: 309
- **Issues:**
  - `CONTAINS_WITH_SUBSTRING_HEAD`: 65

## Summary

| Category | Checked | Flagged | Flag % | Top Issue |
|----------|---------|---------|--------|-----------|
| 1. Hallucination | 2244 triples | 994 | 44.3% | EVIDENCE_NOT_IN_SOURCE |
| 2. Type Misclass | 3028 entities | 389 | 12.8% | LIKELY_MECHANISM_GOT_CONCEPT |
| 3. Omission | 500 chunks | 90 | 18.0% | ZERO_EXTRACTION_DESPITE_CONTENT |
| 4. Name Noise | 3028 entities | 467 | 15.4% | CONTAINS_BRACKETS_OR_DIGITS |