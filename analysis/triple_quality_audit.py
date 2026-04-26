#!/usr/bin/env python3
"""Proportional sampling audit of raw triples for 4 error categories.

Usage:
    python analysis/triple_quality_audit.py [--sample-size N] [--seed SEED]

Outputs:
    - Console report with statistics and examples
    - analysis/sample_results.json: detailed annotated samples
    - analysis/audit_report.md: structured markdown report
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# --- Config ---
RAW_DIR = Path("data/triples/raw")
CHUNKS_DIR = Path("data/chunks")
OUTPUT_DIR = Path("analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENTITY_TYPES = {"protocol", "concept", "mechanism", "metric"}
RELATION_TYPES = {"contains", "depends_on", "compared_with", "applied_to"}

# Known networking protocols (for type detection heuristics)
PROTOCOL_PATTERNS = re.compile(
    r"^(TCP|UDP|IP|HTTP|HTTPS|FTP|SMTP|DNS|DHCP|ARP|RARP|ICMP|IGMP|"
    r"BGP|OSPF|RIP|SNMP|NAT|MPLS|RSVP|SSH|TLS|SSL|POP3|IMAP|"
    r"LTE|5G|4G|3G|2G|WiFi|CSMA|CDMA|TDMA|FDMA|"
    r"Ethernet|Token|Ring|ATM|Frame|Relay|PPP|SLIP"
    r")"
)

# Known suffixes that indicate mechanism type
MECHANISM_KEYWORDS = [
    "算法", "机制", "握手", "控制", "路由", "调度",
    "加密", "认证", "协商", "检测", "纠错", "重传",
    "复用", "交换", "转发", "过滤", "缓存", "压缩",
    "Algorithm", "handshake", "control",
]

# Known metric suffixes
METRIC_SUFFIXES = re.compile(r"(率|时延|带宽|吞吐|RTT|MTU|MSS|QoS|延迟|抖动)$")

# Entity name quality thresholds
MAX_ENTITY_NAME_LENGTH = 30
MIN_ENTITY_NAME_LENGTH = 2
NOISE_PATTERNS = re.compile(r"[(){}\[\]\d]+")


def load_raw_files(raw_dir: Path) -> list[Path]:
    """Load all raw triple JSON files, return sorted paths."""
    files = sorted(raw_dir.glob("*.json"))
    return files


def parse_raw_filename(filename: str) -> tuple[str, int]:
    """Parse '<stem>_chunk_XXXX.json' → (stem, chunk_index)."""
    m = re.match(r"(.+)_chunk_(\d+)\.json$", filename)
    if m:
        return m.group(1), int(m.group(2))
    # fallback: try without chunk suffix
    return Path(filename).stem, -1


def load_source_chunk(stem: str, chunk_index: int) -> dict | None:
    """Find and load the source chunk data from chunks directory."""
    # Try exact stem match first
    chunk_file = CHUNKS_DIR / f"{stem}.json"
    if not chunk_file.exists():
        # Try fuzzy match: stem might have extra suffixes
        candidates = list(CHUNKS_DIR.glob(f"{stem}*.json"))
        if not candidates:
            # Try matching by source_file field
            for cf in CHUNKS_DIR.glob("*.json"):
                try:
                    data = json.loads(cf.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        for c in data:
                            sf = c.get("source_file", "")
                            if stem in sf or sf in stem:
                                chunk_file = cf
                                break
                except Exception:
                    continue
            if not chunk_file.exists() if isinstance(chunk_file, Path) else True:
                return None
        else:
            chunk_file = candidates[0]

    try:
        data = json.loads(chunk_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for c in data:
                if c.get("chunk_index") == chunk_index:
                    return c
            # fallback: try first chunk
            if data:
                return data[0]
    except Exception:
        return None
    return None


# --- Category 1: Hallucination / Evidence Check ---

def check_evidence_support(triple: dict, source_chunk: dict | None) -> dict:
    """Check if evidence text plausibly supports the triple."""
    evidence = triple.get("evidence", "").strip()
    head = triple.get("head", "").strip()
    tail = triple.get("tail", "").strip()
    issues = []

    # Heuristic 1: Empty evidence
    if not evidence or len(evidence) < 5:
        issues.append("EMPTY_EVIDENCE")
        return {"score": 0.0, "issues": issues}

    # Heuristic 2: Evidence doesn't contain head or tail
    head_in_ev = head in evidence
    tail_in_ev = tail in evidence
    if not head_in_ev and not tail_in_ev:
        issues.append("NO_ENTITY_IN_EVIDENCE")
    elif not head_in_ev:
        issues.append("HEAD_NOT_IN_EVIDENCE")
    elif not tail_in_ev:
        issues.append("TAIL_NOT_IN_EVIDENCE")

    # Heuristic 3: Evidence is just a repeat of the triple (no actual text)
    if evidence == f"{head}与{tail}的关系" or evidence == f"{head}和{tail}":
        issues.append("EVIDENCE_IS_GENERIC")

    # Heuristic 4: If source chunk available, check evidence is in source
    if source_chunk:
        content = source_chunk.get("content", "")
        if evidence not in content:
            issues.append("EVIDENCE_NOT_IN_SOURCE")

    # Score: 1.0 = all checks pass, lower = more issues
    score = 1.0
    if "EMPTY_EVIDENCE" in issues:
        score -= 0.5
    if "NO_ENTITY_IN_EVIDENCE" in issues:
        score -= 0.4
    elif "HEAD_NOT_IN_EVIDENCE" in issues or "TAIL_NOT_IN_EVIDENCE" in issues:
        score -= 0.2
    if "EVIDENCE_IS_GENERIC" in issues:
        score -= 0.3
    if "EVIDENCE_NOT_IN_SOURCE" in issues:
        score -= 0.3

    return {"score": max(0.0, score), "issues": issues}


# --- Category 2: Type Misclassification ---

def check_type_misclassification(entity: dict) -> dict:
    """Heuristically check if entity type seems correct."""
    name = entity.get("name", "").strip()
    etype = entity.get("type", "").strip().lower()
    issues = []
    suggested_type = None

    if not name or etype not in ENTITY_TYPES:
        return {"issues": ["INVALID_TYPE"], "suggested": None}

    # Heuristic: Protocol detection
    is_likely_protocol = False
    # Known protocol acronyms
    if PROTOCOL_PATTERNS.match(name.upper()):
        is_likely_protocol = True
    # Chinese protocol naming
    if "协议" in name and len(name) <= 15:
        is_likely_protocol = True

    if is_likely_protocol and etype != "protocol":
        issues.append(f"LIKELY_PROTOCOL_GOT_{etype.upper()}")
        suggested_type = "protocol"

    # Heuristic: Mechanism detection
    is_likely_mechanism = any(kw in name for kw in MECHANISM_KEYWORDS)
    if is_likely_mechanism and etype not in ("mechanism",):
        issues.append(f"LIKELY_MECHANISM_GOT_{etype.upper()}")
        if suggested_type is None:
            suggested_type = "mechanism"

    # Heuristic: Metric detection
    is_likely_metric = bool(METRIC_SUFFIXES.search(name))
    if is_likely_metric and etype != "metric":
        issues.append(f"LIKELY_METRIC_GOT_{etype.upper()}")
        if suggested_type is None:
            suggested_type = "metric"

    return {"issues": issues, "suggested": suggested_type}


# --- Category 3: Obvious Omission (by chunk) ---

def check_chunk_omission(chunk_data: dict) -> dict:
    """Check if a chunk with substantial content has suspiciously few extractions."""
    content = chunk_data.get("content", "").strip()
    entities = chunk_data.get("entities", [])
    triples = chunk_data.get("triples", [])
    issues = []

    content_len = len(content)
    entity_count = len(entities)
    triple_count = len(triples)

    # Chunk has substantial text but NO extractions
    if content_len > 200 and entity_count == 0 and triple_count == 0:
        issues.append("ZERO_EXTRACTION_DESPITE_CONTENT")

    # Chunk has text but only 1-2 extractions (likely under-extracted)
    if content_len > 500 and triple_count <= 2:
        issues.append("LOW_EXTRACTION_RATIO")

    # Chunk has text with protocol names but no protocol entities
    if content_len > 200 and entity_count == 0:
        proto_mentions = len(re.findall(r"\b(TCP|UDP|IP|HTTP|DNS|FTP|SMTP|BGP|OSPF)\b", content, re.IGNORECASE))
        if proto_mentions >= 2:
            issues.append("PROTOCOLS_MENTIONED_BUT_NOT_EXTRACTED")

    return {
        "issues": issues,
        "content_length": content_len,
        "entity_count": entity_count,
        "triple_count": triple_count,
    }


# --- Category 4: Entity Name Inconsistency / Noise ---

def check_entity_name_quality(name: str) -> dict:
    """Check entity name for consistency and noise issues."""
    issues = []

    if not name or not name.strip():
        return {"issues": ["EMPTY_NAME"], "length": 0}

    name = name.strip()
    nlen = len(name)

    # Too short
    if nlen < MIN_ENTITY_NAME_LENGTH:
        issues.append(f"TOO_SHORT({nlen})")

    # Too long (schema max is 30, check at 25 for warning)
    if nlen > MAX_ENTITY_NAME_LENGTH:
        issues.append(f"TOO_LONG({nlen})")

    # Contains brackets/parentheses (noise indicator)
    if NOISE_PATTERNS.search(name):
        issues.append("CONTAINS_BRACKETS_OR_DIGITS")

    # Contains English+Chinese mix (potential noise)
    has_cn = bool(re.search(r"[\u4e00-\u9fff]", name))
    has_en = bool(re.search(r"[a-zA-Z]", name))
    if has_cn and has_en and "（" not in name and "(" not in name:
        # Mixed without explicit parenthetical - could be noise
        # But also could be valid like "TCP协议"
        if not name.endswith(("协议", "机制", "算法", "方法", "技术")):
            issues.append("MIXED_LANG_NO_SUFFIX")

    # Contains punctuation mid-name (possible fragment)
    if re.search(r"[，,。；;]", name):
        issues.append("CONTAINS_PUNCTUATION")

    return {"issues": issues, "length": nlen}


def check_entity_consistency_across_chunk(entities: list[dict]) -> list[dict]:
    """Check for near-duplicate entities within a single chunk."""
    conflicts = []
    names = [e.get("name", "").strip() for e in entities]
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            # Check if one is substring of another
            if n1 in n2 or n2 in n1:
                if len(n1) > 0 and len(n2) > 0 and n1 != n2:
                    conflicts.append({
                        "entity_a": n1,
                        "entity_b": n2,
                        "type": "SUBSTRING_MATCH",
                    })
            # Check if differ only by suffix like 协议
            if n1.rstrip("协议机制算法方法技术") == n2.rstrip("协议机制算法方法技术") and n1 != n2:
                conflicts.append({
                    "entity_a": n1,
                    "entity_b": n2,
                    "type": "SUFFIX_VARIANT",
                })
    return conflicts


# --- Relation Check ---

def check_relation_quality(triple: dict) -> dict:
    """Check relation quality heuristics."""
    relation = triple.get("relation", "").strip().lower()
    head = triple.get("head", "").strip()
    tail = triple.get("tail", "").strip()
    issues = []

    if relation not in RELATION_TYPES:
        issues.append("INVALID_RELATION_TYPE")
        return {"issues": issues}

    # Heuristic: self-relation
    if head == tail:
        issues.append("SELF_REFERENCING")

    # Heuristic: contains should have asymmetric names
    if relation == "contains" and head in tail:
        issues.append("CONTAINS_WITH_SUBSTRING_HEAD")

    return {"issues": issues}


# --- Main Analysis ---

def analyze_sample(
    raw_files: list[Path],
    sample_size: int = 500,
    seed: int = 42,
) -> dict:
    """Sample triples and run all 4 diagnostic categories."""
    random.seed(seed)
    sampled_files = random.sample(raw_files, min(sample_size, len(raw_files)))

    results = {
        "sample_size": len(sampled_files),
        "total_files_in_dir": len(raw_files),
        "total_triples_sampled": 0,
        "total_entities_sampled": 0,
        "category_1_hallucination": {
            "triples_checked": 0,
            "score_distribution": [],
            "issues": Counter(),
            "flagged_count": 0,
            "examples": [],
        },
        "category_2_type_misclass": {
            "entities_checked": 0,
            "type_distribution": Counter(),
            "issues": Counter(),
            "flagged_count": 0,
            "examples": [],
        },
        "category_3_omission": {
            "chunks_checked": 0,
            "issues": Counter(),
            "flagged_count": 0,
            "examples": [],
        },
        "category_4_name_noise": {
            "entities_checked": 0,
            "issues": Counter(),
            "flagged_count": 0,
            "conflicts_found": 0,
            "examples": [],
            "length_distribution": Counter(),
        },
        "relation_quality": {
            "checked": 0,
            "issues": Counter(),
            "relation_distribution": Counter(),
        },
    }

    source_cache = {}

    for fpath in sampled_files:
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            continue

        entities = data.get("entities", [])
        triples = data.get("triples", [])
        stem, chunk_idx = parse_raw_filename(fpath.name)

        # Load source chunk (with caching)
        cache_key = (stem, chunk_idx)
        if cache_key not in source_cache:
            source_cache[cache_key] = load_source_chunk(stem, chunk_idx)
        source_chunk = source_cache[cache_key]

        # --- Cat 3: Omission check (per chunk) ---
        merged_chunk_view = {
            "content": source_chunk.get("content", "") if source_chunk else "",
            "entities": entities,
            "triples": triples,
        }
        chunk_check = check_chunk_omission(merged_chunk_view)
        results["category_3_omission"]["chunks_checked"] += 1
        if chunk_check["issues"]:
            results["category_3_omission"]["flagged_count"] += 1
            for issue in chunk_check["issues"]:
                results["category_3_omission"]["issues"][issue] += 1
            if len(results["category_3_omission"]["examples"]) < 15:
                results["category_3_omission"]["examples"].append({
                    "file": fpath.name,
                    "stem": stem,
                    "chunk_index": chunk_idx,
                    "content_len": chunk_check["content_length"],
                    "entity_count": chunk_check["entity_count"],
                    "triple_count": chunk_check["triple_count"],
                    "issues": chunk_check["issues"],
                    "titles": data.get("titles", []),
                    "content_preview": source_chunk.get("content", "")[:200] if source_chunk else "N/A",
                })

        results["total_triples_sampled"] += len(triples)
        results["total_entities_sampled"] += len(entities)

        # --- Cat 1: Hallucination check (per triple) ---
        for t in triples:
            check = check_evidence_support(t, source_chunk)
            results["category_1_hallucination"]["triples_checked"] += 1
            results["category_1_hallucination"]["score_distribution"].append(check["score"])
            if check["score"] < 0.8:
                results["category_1_hallucination"]["flagged_count"] += 1
                for issue in check["issues"]:
                    results["category_1_hallucination"]["issues"][issue] += 1
                if len(results["category_1_hallucination"]["examples"]) < 15:
                    results["category_1_hallucination"]["examples"].append({
                        "file": fpath.name,
                        "head": t["head"],
                        "relation": t["relation"],
                        "tail": t["tail"],
                        "evidence": t.get("evidence", ""),
                        "score": check["score"],
                        "issues": check["issues"],
                    })

            # Relation quality
            rel_check = check_relation_quality(t)
            results["relation_quality"]["checked"] += 1
            results["relation_quality"]["relation_distribution"][t.get("relation", "unknown")] += 1
            if rel_check["issues"]:
                for issue in rel_check["issues"]:
                    results["relation_quality"]["issues"][issue] += 1

        # --- Cat 2: Type misclassification (per entity) ---
        for e in entities:
            check = check_type_misclassification(e)
            results["category_2_type_misclass"]["entities_checked"] += 1
            results["category_2_type_misclass"]["type_distribution"][e.get("type", "unknown")] += 1
            if check["issues"]:
                results["category_2_type_misclass"]["flagged_count"] += 1
                for issue in check["issues"]:
                    results["category_2_type_misclass"]["issues"][issue] += 1
                if len(results["category_2_type_misclass"]["examples"]) < 20:
                    results["category_2_type_misclass"]["examples"].append({
                        "file": fpath.name,
                        "name": e["name"],
                        "actual_type": e.get("type", "?"),
                        "suggested_type": check["suggested"],
                        "issues": check["issues"],
                    })

        # --- Cat 4: Entity name quality (per entity) ---
        for e in entities:
            name_check = check_entity_name_quality(e.get("name", ""))
            results["category_4_name_noise"]["entities_checked"] += 1
            nlen = name_check["length"]
            # Bucket length
            if nlen <= 5:
                bucket = "1-5"
            elif nlen <= 10:
                bucket = "6-10"
            elif nlen <= 20:
                bucket = "11-20"
            elif nlen <= 30:
                bucket = "21-30"
            else:
                bucket = "31+"
            results["category_4_name_noise"]["length_distribution"][bucket] += 1
            if name_check["issues"]:
                results["category_4_name_noise"]["flagged_count"] += 1
                for issue in name_check["issues"]:
                    results["category_4_name_noise"]["issues"][issue] += 1
                if len(results["category_4_name_noise"]["examples"]) < 20:
                    results["category_4_name_noise"]["examples"].append({
                        "file": fpath.name,
                        "name": e.get("name", ""),
                        "type": e.get("type", "?"),
                        "issues": name_check["issues"],
                    })

        # --- Cat 4: Intra-chunk consistency ---
        conflicts = check_entity_consistency_across_chunk(entities)
        if conflicts:
            results["category_4_name_noise"]["conflicts_found"] += len(conflicts)
            if len(results["category_4_name_noise"]["examples"]) < 20:
                for c in conflicts[:3]:
                    results["category_4_name_noise"]["examples"].append({
                        "file": fpath.name,
                        "entity_a": c["entity_a"],
                        "entity_b": c["entity_b"],
                        "conflict_type": c["type"],
                    })

    return results


def format_report(results: dict) -> str:
    """Generate a structured markdown report from analysis results."""
    lines = []
    lines.append("# Raw Triple Quality Audit Report")
    lines.append("")
    lines.append(f"**Sample size:** {results['sample_size']} files out of {results['total_files_in_dir']} total")
    lines.append(f"**Triples sampled:** {results['total_triples_sampled']}")
    lines.append(f"**Entities sampled:** {results['total_entities_sampled']}")
    lines.append("")

    # --- Category 1 ---
    cat1 = results["category_1_hallucination"]
    lines.append("## Category 1: 幻觉 / Evidence 不支撑")
    lines.append("")
    checked = cat1["triples_checked"]
    flagged = cat1["flagged_count"]
    pct = (flagged / checked * 100) if checked else 0
    lines.append(f"- **Triples checked:** {checked}")
    lines.append(f"- **Flagged (score < 0.8):** {flagged} ({pct:.1f}%)")
    if cat1["score_distribution"]:
        avg_score = sum(cat1["score_distribution"]) / len(cat1["score_distribution"])
        lines.append(f"- **Average evidence score:** {avg_score:.3f}")
    lines.append(f"- **Issue breakdown:**")
    for issue, count in cat1["issues"].most_common():
        lines.append(f"  - `{issue}`: {count}")
    if cat1["examples"]:
        lines.append("")
        lines.append("### Flagged Examples (Hallucination)")
        lines.append("")
        for ex in cat1["examples"][:10]:
            lines.append(f"- **File:** `{ex['file']}`")
            lines.append(f"  - Triple: ({ex['head']}) -[{ex['relation']}]-> ({ex['tail']})")
            lines.append(f"  - Evidence: {ex['evidence'][:150]}")
            lines.append(f"  - Score: {ex['score']:.2f} | Issues: {ex['issues']}")
            lines.append("")
    lines.append("")

    # --- Category 2 ---
    cat2 = results["category_2_type_misclass"]
    lines.append("## Category 2: 类型选错")
    lines.append("")
    checked_e = cat2["entities_checked"]
    flagged_e = cat2["flagged_count"]
    pct_e = (flagged_e / checked_e * 100) if checked_e else 0
    lines.append(f"- **Entities checked:** {checked_e}")
    lines.append(f"- **Flagged (suspicious type):** {flagged_e} ({pct_e:.1f}%)")
    lines.append(f"- **Type distribution in sample:**")
    for etype, count in cat2["type_distribution"].most_common():
        lines.append(f"  - `{etype}`: {count}")
    lines.append(f"- **Issue breakdown:**")
    for issue, count in cat2["issues"].most_common():
        lines.append(f"  - `{issue}`: {count}")
    if cat2["examples"]:
        lines.append("")
        lines.append("### Flagged Examples (Type Mismatch)")
        lines.append("")
        for ex in cat2["examples"][:10]:
            lines.append(f"- `{ex['name']}`: actual=`{ex['actual_type']}` → suggested=`{ex['suggested_type']}` ({ex['issues']})")
        lines.append("")
    lines.append("")

    # --- Category 3 ---
    cat3 = results["category_3_omission"]
    lines.append("## Category 3: 明显遗漏")
    lines.append("")
    chunks_c = cat3["chunks_checked"]
    flagged_c = cat3["flagged_count"]
    pct_c = (flagged_c / chunks_c * 100) if chunks_c else 0
    lines.append(f"- **Chunks checked:** {chunks_c}")
    lines.append(f"- **Flagged (suspicious omission):** {flagged_c} ({pct_c:.1f}%)")
    lines.append(f"- **Issue breakdown:**")
    for issue, count in cat3["issues"].most_common():
        lines.append(f"  - `{issue}`: {count}")
    if cat3["examples"]:
        lines.append("")
        lines.append("### Flagged Examples (Omission)")
        lines.append("")
        for ex in cat3["examples"][:10]:
            lines.append(f"- **File:** `{ex['file']}`")
            lines.append(f"  - Titles: {ex.get('titles', [])}")
            lines.append(f"  - Content length: {ex['content_len']} | Entities: {ex['entity_count']} | Triples: {ex['triple_count']}")
            lines.append(f"  - Issues: {ex['issues']}")
            lines.append(f"  - Content preview: {ex.get('content_preview', 'N/A')[:150]}")
            lines.append("")
    lines.append("")

    # --- Category 4 ---
    cat4 = results["category_4_name_noise"]
    lines.append("## Category 4: 实体名不一致 / 噪声")
    lines.append("")
    checked_n = cat4["entities_checked"]
    flagged_n = cat4["flagged_count"]
    pct_n = (flagged_n / checked_n * 100) if checked_n else 0
    lines.append(f"- **Entities checked:** {checked_n}")
    lines.append(f"- **Flagged (name quality issue):** {flagged_n} ({pct_n:.1f}%)")
    lines.append(f"- **Intra-chunk conflicts (near-duplicates):** {cat4['conflicts_found']}")
    lines.append(f"- **Name length distribution:**")
    for bucket in ["1-5", "6-10", "11-20", "21-30", "31+"]:
        count = cat4["length_distribution"].get(bucket, 0)
        lines.append(f"  - `{bucket}` chars: {count}")
    lines.append(f"- **Issue breakdown:**")
    for issue, count in cat4["issues"].most_common():
        lines.append(f"  - `{issue}`: {count}")
    if cat4["examples"]:
        lines.append("")
        lines.append("### Flagged Examples (Name Quality)")
        lines.append("")
        for ex in cat4["examples"][:15]:
            lines.append(f"- `{ex.get('name', ex.get('entity_a', '?'))}`: {ex.get('issues', ex.get('conflict_type', '?'))}")
        lines.append("")
    lines.append("")

    # --- Relation Quality ---
    rel = results["relation_quality"]
    lines.append("## Relation Quality")
    lines.append("")
    lines.append(f"- **Triples checked:** {rel['checked']}")
    lines.append(f"- **Relation distribution:**")
    for rtype, count in rel["relation_distribution"].most_common():
        lines.append(f"  - `{rtype}`: {count}")
    if rel["issues"]:
        lines.append(f"- **Issues:**")
        for issue, count in rel["issues"].most_common():
            lines.append(f"  - `{issue}`: {count}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Category | Checked | Flagged | Flag % | Top Issue |")
    lines.append("|----------|---------|---------|--------|-----------|")
    lines.append(f"| 1. Hallucination | {checked} triples | {flagged} | {pct:.1f}% | {cat1['issues'].most_common(1)[0][0] if cat1['issues'] else 'N/A'} |")
    lines.append(f"| 2. Type Misclass | {checked_e} entities | {flagged_e} | {pct_e:.1f}% | {cat2['issues'].most_common(1)[0][0] if cat2['issues'] else 'N/A'} |")
    lines.append(f"| 3. Omission | {chunks_c} chunks | {flagged_c} | {pct_c:.1f}% | {cat3['issues'].most_common(1)[0][0] if cat3['issues'] else 'N/A'} |")
    lines.append(f"| 4. Name Noise | {checked_n} entities | {flagged_n} | {pct_n:.1f}% | {cat4['issues'].most_common(1)[0][0] if cat4['issues'] else 'N/A'} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit raw triples for quality issues")
    parser.add_argument("--sample-size", type=int, default=500, help="Number of files to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--raw-dir", type=str, default="data/triples/raw", help="Raw triples directory")
    parser.add_argument("--all", action="store_true", help="Process ALL files (not sample)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print(f"ERROR: Directory not found: {raw_dir}")
        sys.exit(1)

    raw_files = load_raw_files(raw_dir)
    print(f"Found {len(raw_files)} raw triple files in {raw_dir}")

    sample_size = len(raw_files) if args.all else min(args.sample_size, len(raw_files))
    print(f"Sampling {sample_size} files (seed={args.seed})...")

    results = analyze_sample(raw_files, sample_size=sample_size, seed=args.seed)

    # Save detailed JSON
    json_path = OUTPUT_DIR / "sample_results.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Detailed results saved to {json_path}")

    # Generate and save report
    report = format_report(results)
    report_path = OUTPUT_DIR / "audit_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to {report_path}")

    # Print summary to console
    print("\n" + "=" * 70)
    cat1 = results["category_1_hallucination"]
    cat2 = results["category_2_type_misclass"]
    cat3 = results["category_3_omission"]
    cat4 = results["category_4_name_noise"]

    c1 = cat1["triples_checked"]
    c2 = cat2["entities_checked"]
    c3 = cat3["chunks_checked"]
    c4 = cat4["entities_checked"]

    print(f"Sample: {results['sample_size']} files, {results['total_triples_sampled']} triples, {results['total_entities_sampled']} entities")
    print()
    print(f"1. Hallucination:  {cat1['flagged_count']}/{c1} flagged ({cat1['flagged_count']/c1*100:.1f}%)" if c1 else "1. Hallucination: N/A")
    top1 = cat1["issues"].most_common(3) if cat1["issues"] else []
    for issue, count in top1:
        print(f"   - {issue}: {count}")

    print(f"2. Type Misclass:  {cat2['flagged_count']}/{c2} flagged ({cat2['flagged_count']/c2*100:.1f}%)" if c2 else "2. Type Misclass: N/A")
    top2 = cat2["issues"].most_common(3) if cat2["issues"] else []
    for issue, count in top2:
        print(f"   - {issue}: {count}")

    print(f"3. Omission:       {cat3['flagged_count']}/{c3} chunks ({cat3['flagged_count']/c3*100:.1f}%)" if c3 else "3. Omission: N/A")
    top3 = cat3["issues"].most_common(3) if cat3["issues"] else []
    for issue, count in top3:
        print(f"   - {issue}: {count}")

    print(f"4. Name Noise:     {cat4['flagged_count']}/{c4} entities ({cat4['flagged_count']/c4*100:.1f}%)" if c4 else "4. Name Noise: N/A")
    top4 = cat4["issues"].most_common(3) if cat4["issues"] else []
    for issue, count in top4:
        print(f"   - {issue}: {count}")

    print()
    print("=" * 70)
    print(f"Full report: {report_path}")
    print(f"JSON details: {json_path}")


if __name__ == "__main__":
    main()
