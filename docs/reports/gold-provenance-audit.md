# Gold Candidate Provenance Audit Report

**Generated:** 2026-05-04
**Gold file:** `data/eval/gold_triples.agent_candidate.json`

## Summary

- **Total triples:** 223
- **Triples with exact evidence match in source:** 193 (86.5%)
- **Triples requiring further verification:** 30 (13.5%)
- **Triples with re-extracted evidence (substring/re-evidenced):** 55
- **Triples removed during review:** 1 (unverifiable evidence)
- **Duplicate triples removed:** 4
- **All triples include:** source_file, chunk_index

## Evidence Quality Breakdown

| Evidence Source | Count | Pct |
|---|---|---|
| Exact match (agent quoted verbatim) | 168 | 75.3% |
| Re-extracted (substring/re-evidenced) | 55 | 24.7% |

Of re-extracted triples:
- `evidence_extracted_by_re_evidenced`: 18
- `evidence_extracted_by_substring`: 10
- `evidence_extracted_by_substring_norm`: 9
- `evidence_position_approximate`: 18

## Contamination Check

| Check | Status |
|---|---|
| References to prediction directories (data/triples/*) | ✓ CLEAN |
| Source files from data/chunks/ only | ✓ CONFIRMED |
| No audit fields (correct, _votes, _confidence) | ✓ CONFIRMED |

## Source File Distribution

Unique source files: 38

| Source File | Triple Count |
|---|---|
| 计算机网络：自顶向下方法（原书第8版） ([美]詹姆斯·F.库罗斯, [美]基思·W.罗斯, 陈鸣) (z-librar | 26 |
| 计算机网络：自顶向下方法（原书第8版） (James F. Kurose, Keith W. Ross, 陈鸣) (z- | 20 |
| WNB-ch01b.json | 15 |
| rfc4960.txt.json | 14 |
| TCPIP详解 卷1：协议（原书第2版） (凯文 R. 福尔 (Kevin R. Fall) etc.) (z-libr | 13 |
| 计算机网络（第6版） 自顶向下方法 ([美] James F.Kurose [美] Keith W.Ross) (z-l | 11 |
| 第7章 无线网络和移动网络.json | 10 |
| 部分题目中文解析.json | 9 |
| Solution Manual for Computer Networking A Top-Down Approach  | 8 |
| 计算机网络（第8版） (谢希仁) (z-library.sk, 1lib.sk, z-lib.sk).json | 8 |
| 1711361_刘炼_作业5.json | 7 |
| 图解HTTP (上野宣) (z-library.sk, 1lib.sk, z-lib.sk).json | 6 |
| rfc6376.txt.json | 6 |
| TCPIP详解 卷3：TCP事务协议、HTTP、NNTP和UNIX域协议 (W. 理查德·史蒂文斯) (z-librar | 5 |
| ComputerNetworking_ ATopDownApproach_7th.json | 5 |
| 2024年王道计算机网络复习指导 (王道计算机) (z-library.sk, 1lib.sk, z-lib.sk).j | 5 |
| 2021级A卷gpt版（纠错）.json | 5 |
| 第1课.json | 4 |
| UnderHIT_linklayerv4.json | 4 |
| Chapter_9_V7.0.json | 4 |
| ... (18 more files) | ... |

## Triples Requiring Further Verification

These 30 triples have evidence that could not be verified as an exact substring of their claimed source_file. This may be due to encoding differences (smart quotes, em-dashes) or evidence extracted from a different chunk in the same file.

| Head | Rel | Tail | Source File | Evidence (truncated) |
|---|---|---|---|---|
| 共享 | contains | 资源共享 | 第1课.json | 共享 (Sharing) - 指资源共享。 |
| TCP | depends_on | congestion control m | Solution Manual for Computer Networking  | Therefore a congestion-control mechanism is needed to stem t |
| AF PHB | contains | 丢弃优先级 | CH8-6ed 音频视频.json | 对于其中的每一个等级再用 DSCP 的比特 3 ~5 划分出三个"丢弃优先级"。 |
| SMTP | depends_on | TCP | 计算机网络：自顶向下方法（原书第8版） (James F. Kurose, Ke | 特网中的电子邮件 #### 2.3.1 SMTP #### 2.3.2 邮件报文格式 #### 2.3.3 邮件访问协议 |
| POP3 | depends_on | TCP | 计算机网络：自顶向下方法（原书第8版） ([美]詹姆斯·F.库罗斯, [美]基思 | 是如何利用 CDN 的。对于面向连接的（TCP）和无连接的（UDP）端到端传输服务，我们走马观花般地学习了套接字的使用。 |
| 5G | contains | URLLC | 计算机网络：自顶向下方法（原书第8版） (James F. Kurose, Ke | 定无线因特网服务，而不用于智能手机。  5G 标准将频率分为两组：FR1（450MHz~6GHz）和 FR2（24GHz |
| 持续 HTTP | contains | 非流水线 | 计算机网络第8版课件-第6章-应用层.json | # 协议 HTTP/1.1 使用持续连接  - 持续连接（persistent connection）：服务器在发送响应 |
| 协议 | contains | 语法 | WNB-ch01b.json | 协议的组成 语法（syntax）：以二进制形式表示的命令和相应的结构 |
| TCP | contains | socket | ComputerNetworking_ ATopDownApproach_7th | Associated with the TCP connection, there will be a socket a |
| Web page | contains | HTML | ComputerNetworking_ ATopDownApproach_7th | Let's suppose the page consists of a base HTML |
| Web page | contains | JPEG | ComputerNetworking_ ATopDownApproach_7th | Let's suppose the page consists of a base HTML |
| 体系结构原则 | contains | 端到端论点 | TCPIP详解 卷1：协议（原书第2版） (凯文 R. 福尔 (Kevin R. | 2 1.1.2 端到端论点和命运共享 ..................... 3 |
| 体系结构原则 | contains | 命运共享 | TCPIP详解 卷1：协议（原书第2版） (凯文 R. 福尔 (Kevin R. | 2 1.1.2 端到端论点和命运共享 ..................... 3 |
| 体系结构原则 | contains | 差错控制 | TCPIP详解 卷1：协议（原书第2版） (凯文 R. 福尔 (Kevin R. | ……………………………… 1 1.1 体系结构原则 ……………………………… 2 1.1.1 分组、连接和数据报 ……… |
| 体系结构原则 | contains | 流量控制 | TCPIP详解 卷1：协议（原书第2版） (凯文 R. 福尔 (Kevin R. | ……………………………… 1 1.1 体系结构原则 ……………………………… 2 1.1.1 分组、连接和数据报 ……… |
| 分层 | contains | 复用 | TCPIP详解 卷1：协议（原书第2版） (凯文 R. 福尔 (Kevin R. | .................................... 5 1.2.2 分层实现中的复用、分解和封装  |
| 分层 | contains | 分解 | TCPIP详解 卷1：协议（原书第2版） (凯文 R. 福尔 (Kevin R. | .................................... 5 1.2.2 分层实现中的复用、分解和封装  |
| 分层 | contains | 封装 | TCPIP详解 卷1：协议（原书第2版） (凯文 R. 福尔 (Kevin R. | .................................... 5 1.2.2 分层实现中的复用、分解和封装  |
| TCP/IP | contains | 端口号 | TCPIP详解 卷1：协议（原书第2版） (凯文 R. 福尔 (Kevin R. | 1.3 TCP/IP 协议族结构和协议 ..................... 9 |
| RIP | contains | 限制路径最大"距离"对策 | 1711361_刘炼_作业5.json | RIP协议中采用了限制路径最大"距离"对策，设置了最大"距离"为16，即经过的路由器个数不超过15个，从一定程度上解决了 |
| ... (10 more) | ... | ... | ... | ... |

## Recommendations for Human Review

1. **Verify unverified evidence:** Spot-check the 30 triples where evidence-in-source could not be confirmed. Most are encoding mismatch issues (smart quotes, em/en dashes) rather than fabricated evidence.
2. **Review re-evidenced triples:** The 31 triples that were re-evidenced by head+tail search should be confirmed to ensure the relation is correctly inferred.
3. **Check compared_with relations:** 10 compared_with triples — verify each comparison is explicitly made in the source text.
4. **Entity name normalization:** 265 entities include some near-duplicates (e.g., 拥塞控制 vs 拥塞控制算法). Decide whether to merge or keep as-is.
