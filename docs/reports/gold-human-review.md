# Gold Candidate Human Review

**Reviewed:** 2026-05-05
**Input:** `data/eval/gold_triples.agent_candidate.json`
**Output:** `data/eval/gold_triples.reviewed.json`
**Evaluation gold at review time:** `data/eval/gold_triples.json`

> Note: this report documents the earlier 203-triple reviewed seed set. On 2026-05-06
> the canonical `data/eval/gold_triples.json` was replaced by the verified 500-triple
> gold file. Use `docs/reports/gold-500-provenance-audit.md` and
> `docs/reports/zs-fs-evaluation-500.md` for the current zs/fs evaluation basis.

## Summary

- Input triples: 223
- Accepted triples: 203
- Removed triples: 20
- Corrected triples: 19
- Re-derived entities: 248

## Review Policy

- Accepted only triples whose evidence and source chunk support the head, relation, and tail.
- Removed triples with table-of-contents-only evidence when the entity was too generic or the relation was not explicit.
- Removed triples with wrong relation semantics, especially `depends_on` when the source explicitly said the dependency was optional.
- Corrected evidence strings when the source chunk supported the relation but the candidate quote was truncated or used normalized punctuation.
- Kept original source names unless a quote mark or entity type was clearly inconsistent with the source text.

## Distribution After Review

| Relation | Count |
|---|---:|
| `applied_to` | 8 |
| `compared_with` | 9 |
| `contains` | 117 |
| `depends_on` | 69 |

| Entity Type | Count |
|---|---:|
| `concept` | 127 |
| `mechanism` | 61 |
| `metric` | 16 |
| `protocol` | 44 |

## Removed Triples

| Original Index | Triple | Reason |
|---:|---|---|
| 0 | `TLS (protocol) contains variant (concept)` | 证据只说明 TLS 结构语言中的 structures/variants，不能直接支撑 TLS contains variant。 |
| 2 | `共享 (concept) contains 资源共享 (concept)` | “共享”与“资源共享”在证据中是定义/同义关系，不是 contains。 |
| 44 | `Rdt 2.1 (protocol) contains 状态 (concept)` | tail “状态”过于泛化，证据“状态数量翻倍”不足以形成稳定 KG 实体。 |
| 48 | `SMTP (protocol) depends_on TCP (protocol)` | 证据是章节目录，未说明 SMTP 依赖 TCP。 |
| 49 | `POP3 (protocol) depends_on TCP (protocol)` | 证据只概述应用层协议和 TCP/UDP 服务模型，未说明 POP3 依赖 TCP。 |
| 84 | `SIP (protocol) depends_on RTP (protocol)` | 证据明确说 SIP works with RTP, but does not mandate it，不支持 depends_on。 |
| 85 | `HTTP (protocol) depends_on IP (protocol)` | 证据是解题步骤中获取 IP 地址，不能支撑 HTTP depends_on IP 协议关系。 |
| 97 | `TCP (protocol) contains socket (concept)` | 证据说明 TCP connection 与 socket associated，不支持 TCP protocol contains socket。 |
| 112 | `connection 1 (concept) compared_with connection 2 (concept)` | connection 1/connection 2 是题目图中的编号实体，非稳定课程 KG 实体。 |
| 134 | `体系结构原则 (concept) contains 分组 (concept)` | tail “分组”过于泛化，且证据只是目录层级。 |
| 143 | `TCP/IP (protocol) contains DNS (protocol)` | 证据讨论 iMessage/WhatsApp/SMS 与 TCP/IP 网络，不能支撑 TCP/IP contains DNS。 |
| 158 | `(S,G,rpt) downstream state machine (mechanism) contains Expiry Timer (mechanism)` | 证据为 RFC 首页元数据，未出现 Expiry Timer 或 state machine。 |
| 159 | `(S,G,rpt) downstream state machine (mechanism) contains Prune-Pending Timer (mechanism)` | 证据为 RFC 首页元数据，未出现 Prune-Pending Timer 或 state machine。 |
| 175 | `SCTP (protocol) depends_on rwnd (concept)` | 声明证据不在该 chunk 中；该 chunk 仅出现 initial a_rwnd，不能支撑 SCTP depends_on rwnd。 |
| 177 | `TCP (protocol) contains TCP throughput equation (concept)` | 证据只是 “3.3 TCP/IP 参考模型”，不支持 TCP contains TCP throughput equation。 |
| 185 | `拒绝服务 (mechanism) applied_to 资源 (concept)` | 证据说明 SYN 洪泛导致资源耗尽，但不支持 “拒绝服务 applied_to 资源” 这个关系方向。 |
| 195 | `拥塞窗口 (concept) depends_on RTT (metric)` | 证据是 RTT/TimeoutInterval 计算表，未支撑 拥塞窗口 depends_on RTT。 |
| 200 | `基于连接的服务 (concept) contains 连接 (concept)` | “基于连接的服务 contains 连接”是近似同义/泛化实体，信息量过低。 |
| 216 | `转发器 (mechanism) applied_to 以太网 (protocol)` | 证据讨论物理媒介，未支撑 转发器 applied_to 以太网。 |
| 222 | `DKIM (protocol) applied_to SMTP (protocol)` | 证据只说明使用 SMTP server 的场景，未支撑 DKIM applied_to SMTP。 |

## Corrected Triples

| Original Index | Change |
|---:|---|
| 25 | `evidence`: `Therefore a congestion-control mechanism is needed to stem the flow of "data ...` -> `Therefore a congestion-control mechanism is needed to stem the flow of “data ...` |
| 28 | `evidence`: `对于其中的每一个等级再用 DSCP 的比特 3 ~5 划分出三个"丢弃优先级"。` -> `对于其中的每一个等级再用 DSCP 的比特 3 ~5 划分出三个“丢弃优先级”。` |
| 38 | `evidence`: `无线与有线 LAN 帧地址部分的区别...` -> `802.3 帧  目的 地址 R1 MAC 地址  源地址 H1 MAC 地址  802.11 帧  地址 1 AP MAC 地址  地址 2 H1 MA...` |
| 41 | `evidence`: `Rdt 2.1 vs. Rdt 2.0...` -> `# Rdt 2.1 vs. Rdt 2.0  ## 发送方： - 为每个分组增加了序列号 - 两个序列号 (0, 1) 就够用，为什么？ - 需校验 AC...` |
| 70 | `tail_type`: `protocol` -> `mechanism` |
| 76 | `evidence`: `定无线因特网服务，而不用于智能手机。  5G 标准将频率分为两组：FR1（450MHz~6GHz）和 FR2（24GHz~52GHz）。大多数早期的部署将...` -> `5G 不是一个密切结合的标准，而是由三个共存的标准组成 [Dahlman 2018]：  - eMBB（增强型移动带宽）。5G NR 的初始部署主要集中在...` |
| 78 | `chunk_index`: `239` -> `231`; `evidence`: `然而，如表 7-1 所示，这些标准在物理层有一些重要的区别。802.11 设备工作在两个不同的频段上：2.4GHz~2.485GHz（称为 2.4GHz ...` -> `5G 没有增加功率，而是使用 MIMO 技术（与我们在 7.3 节研究 802.11 网络时遇到的技术相同），该技术在每个基站使用多个天线。每个 MIMO...`; `source_file`: `计算机网络：自顶向下方法（原书第8版） ([美]詹姆斯·F.库罗斯, [美]基思·W.罗斯, 陈鸣) (z-library.sk, 1lib.sk, z-lib.sk).json` -> `计算机网络：自顶向下方法（原书第8版） (James F. Kurose, Keith W. Ross, 陈鸣) (z-library.sk, 1lib.sk, z-lib.sk).json` |
| 79 | `head_type`: `protocol` -> `mechanism` |
| 88 | `evidence`: `# 协议 HTTP/1.1 使用持续连接  - 持续连接（persistent connection）：服务器在发送响应后仍然在一段时间内保持这条连接（不...` -> `两种工作方式：   - 非流水线方式 (without pipelining)   - 流水线方式 (with pipelining)。...` |
| 99 | `evidence`: `Let's suppose the page consists of a base HTML...` -> `Let’s suppose the page consists of a base HTML  2.2 • THE WEB AND HTTP 129  f...` |
| 100 | `evidence`: `Let's suppose the page consists of a base HTML...` -> `Let’s suppose the page consists of a base HTML  2.2 • THE WEB AND HTTP 129  f...` |
| 152 | `evidence`: `RIP协议中采用了限制路径最大"距离"对策，设置了最大"距离"为16，即经过的路由器个数不超过15个，从一定程度上解决了计数到无穷问题时的局限性。` -> `RIP协议中采用了限制路径最大“距离”对策，设置了最大“距离”为16，即经过的路由器个数不超过15个，从一定程度上解决了计数到无穷问题时的局限性。`; `tail`: `限制路径最大"距离"对策` -> `限制路径最大“距离”对策` |
| 156 | `head_type`: `protocol` -> `mechanism` |
| 178 | `evidence`: `From the TCP throughput equation` -> `From the TCP throughput equation $B = \frac{1.22 \cdot MSS}{RTT \cdot \sqrt{L}}$` |
| 179 | `evidence`: `From the TCP throughput equation` -> `From the TCP throughput equation $B = \frac{1.22 \cdot MSS}{RTT \cdot \sqrt{L}}$` |
| 180 | `evidence`: `From the TCP throughput equation` -> `From the TCP throughput equation $B = \frac{1.22 \cdot MSS}{RTT \cdot \sqrt{L}}$` |
| 201 | `evidence`: `算机网络的体系结构（10）  服务分类和服务原语（primitives）  基于连接的服务和无连接服务  基于连接的服务 - 当使用服务传送数据时，首先建...` -> `服务原语可分为四种类型  - 请求（Request）: An entity wants the service to do some work - 指示（...` |
| 205 | `evidence`: `类和服务原语（primitives）  基于连接的服务和无连接服务  基于连接的服务 - 当使用服务传送数...` -> `基于连接的服务 - 当使用服务传送数据时，首先建立连接，然后使用该连接传送数据。使用完后，关闭连接。 - 特点：顺序性好。  无连接服务 - 直接使用服务...` |
| 220 | `evidence`: `If the device is equipped with a private...` -> `If the device is equipped with a private key registered for jdoe@example.com ...` |
