"""Merge, normalize and deduplicate extracted triples."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from .kg_schema import Entity, Triple

logger = logging.getLogger(__name__)

SUFFIX_PATTERN = re.compile(r"(协议|算法|机制|方法|技术|方式)$")

ALIAS_MAP: dict[str, str] = {
    "传输控制协议": "TCP",
    "用户数据报协议": "UDP",
    "超文本传输协议": "HTTP",
    "超文本传送协议": "HTTP",
    "域名系统": "DNS",
    "文件传输协议": "FTP",
    "简单邮件传输协议": "SMTP",
    "简单邮件传送协议": "SMTP",
    "地址解析协议": "ARP",
    "逆地址解析协议": "RARP",
    "网际控制报文协议": "ICMP",
    "网际协议": "IP",
    "往返时延": "RTT",
    "往返时间": "RTT",
    "最大传输单元": "MTU",
    "最大报文段长度": "MSS",
    "服务质量": "QoS",
    "开放系统互连": "OSI",
    "网络地址转换": "NAT",
    "动态主机配置协议": "DHCP",
    "简单网络管理协议": "SNMP",
    "资源预留协议": "RSVP",
    "多协议标签交换": "MPLS",
    "边界网关协议": "BGP",
    "内部网关协议": "IGP",
    "开放最短路径优先": "OSPF",
    "路由信息协议": "RIP",
}


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name to a canonical form."""
    name = name.strip()
    if name in ALIAS_MAP:
        return ALIAS_MAP[name]
    without_suffix = SUFFIX_PATTERN.sub("", name)
    if len(without_suffix) >= 2 and without_suffix in ALIAS_MAP:
        return ALIAS_MAP[without_suffix]
    if without_suffix != name and len(without_suffix) >= 2:
        return without_suffix
    return name


class KGMerger:
    """Merge raw extraction results into a deduplicated knowledge graph."""

    def __init__(self, alias_map: dict[str, str] | None = None):
        if alias_map:
            ALIAS_MAP.update(alias_map)

    def merge_directory(self, raw_dir: Path, output_path: Path) -> Path:
        """Load all raw triple files from a directory and merge them."""
        raw_dir = Path(raw_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        raw_files = sorted(raw_dir.glob("*.json"))
        if not raw_files:
            logger.warning("No raw triple files found in %s", raw_dir)

        all_entities: list[Entity] = []
        all_triples: list[Triple] = []

        for f in tqdm(
            raw_files,
            desc="kgmerge",
            unit="file",
            total=len(raw_files),
            dynamic_ncols=True,
        ):
            data = json.loads(f.read_text(encoding="utf-8"))
            for e in data.get("entities", []):
                all_entities.append(
                    Entity(
                        name=e["name"],
                        type=e["type"],
                        description=e.get("description", ""),
                    )
                )
            for t in data.get("triples", []):
                all_triples.append(
                    Triple(
                        head=t["head"],
                        relation=t["relation"],
                        tail=t["tail"],
                        evidence=t.get("evidence", ""),
                    )
                )

        merged_entities = self._merge_entities(all_entities)
        entity_names = {e.name for e in merged_entities}
        merged_triples = self._merge_triples(all_triples, entity_names)

        entity_type_map = {e.name: e.type for e in merged_entities}

        result = {
            "entities": [
                {
                    "name": e.name,
                    "type": e.type,
                    "description": e.description,
                    "frequency": e.frequency,
                }
                for e in sorted(merged_entities, key=lambda x: x.frequency, reverse=True)
            ],
            "triples": [
                {
                    "head": t.head,
                    "head_type": entity_type_map.get(t.head, ""),
                    "relation": t.relation,
                    "tail": t.tail,
                    "tail_type": entity_type_map.get(t.tail, ""),
                    "frequency": t.frequency,
                    "evidences": t.evidences,
                }
                for t in sorted(merged_triples, key=lambda x: x.frequency, reverse=True)
            ],
            "stats": self._compute_stats(merged_entities, merged_triples),
        }

        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "Merged: %d entities, %d triples -> %s",
            len(merged_entities),
            len(merged_triples),
            output_path,
        )
        return output_path

    def _merge_entities(self, entities: list[Entity]) -> list[Entity]:
        """Normalize names, deduplicate, accumulate frequency."""
        grouped: dict[str, Entity] = {}
        for e in entities:
            if not e.is_valid():
                continue
            canonical = normalize_entity_name(e.name)
            if len(canonical) < 2:
                continue
            if canonical in grouped:
                grouped[canonical].frequency += 1
                if not grouped[canonical].description and e.description:
                    grouped[canonical].description = e.description
            else:
                grouped[canonical] = Entity(
                    name=canonical,
                    type=e.type,
                    description=e.description,
                    frequency=1,
                )
        return list(grouped.values())

    def _merge_triples(
        self, triples: list[Triple], valid_entity_names: set[str]
    ) -> list[Triple]:
        """Normalize head/tail, deduplicate, accumulate frequency and evidences."""
        grouped: dict[tuple[str, str, str], Triple] = {}
        for t in triples:
            if not t.is_valid():
                continue
            head = normalize_entity_name(t.head)
            tail = normalize_entity_name(t.tail)
            if head == tail:
                continue
            if head not in valid_entity_names or tail not in valid_entity_names:
                continue
            key = (head, t.relation, tail)
            if key in grouped:
                grouped[key].frequency += 1
                if t.evidence and t.evidence not in grouped[key].evidences:
                    grouped[key].evidences.append(t.evidence)
            else:
                grouped[key] = Triple(
                    head=head,
                    relation=t.relation,
                    tail=tail,
                    frequency=1,
                    evidences=[t.evidence] if t.evidence else [],
                )
        return list(grouped.values())

    @staticmethod
    def _compute_stats(entities: list[Entity], triples: list[Triple]) -> dict:
        entities_by_type: dict[str, int] = defaultdict(int)
        for e in entities:
            entities_by_type[e.type] += 1
        triples_by_relation: dict[str, int] = defaultdict(int)
        for t in triples:
            triples_by_relation[t.relation] += 1
        return {
            "total_entities": len(entities),
            "total_triples": len(triples),
            "entities_by_type": dict(entities_by_type),
            "triples_by_relation": dict(triples_by_relation),
        }
