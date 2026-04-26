#!/usr/bin/env python3
"""
Sampling script for generating a structured gold-triples template.
Reads data/triples/merged/merged_triples.json and outputs data/eval/gold_triples.json
with 60 stratified-random samples across (head_type, relation) strata.
"""

import json
import random
from pathlib import Path
from typing import Any


def load_triples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "triples" in data and isinstance(data["triples"], list):
        return data["triples"]
    if isinstance(data, list):
        return data
    return []


def first_evidence(triple: dict[str, Any]) -> str:
    evidences = triple.get("evidences")
    if isinstance(evidences, list) and evidences:
        e = evidences[0]
        return str(e) if isinstance(e, str) else str(e)
    ev = triple.get("evidence", "")
    return str(ev) if isinstance(ev, str) else ""


def build_strata(
    triples: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for t in triples:
        head_type = t.get("head_type", "unknown")
        relation = t.get("relation", "unknown")
        key = (head_type, relation)
        strata.setdefault(key, []).append(t)
    return strata


def compute_quotas(
    strata: dict[tuple[str, str], list[dict[str, Any]]], total: int
) -> dict[tuple[str, str], int]:
    total_items = sum(len(items) for items in strata.values()) or 1
    quotas: dict[tuple[str, str], int] = {}
    for key, items in strata.items():
        q = max(1, round(total * len(items) / total_items))
        quotas[key] = min(q, len(items))

    # Adjust to exact total
    quota_sum = sum(quotas.values())
    if quota_sum < total:
        deficit = total - quota_sum
        ordered = sorted(strata.items(), key=lambda kv: len(kv[1]), reverse=True)
        while deficit > 0:
            for key, items in ordered:
                if quotas[key] < len(items):
                    quotas[key] += 1
                    deficit -= 1
                    if deficit == 0:
                        break
    elif quota_sum > total:
        excess = quota_sum - total
        ordered = sorted(quotas.keys(), key=lambda k: quotas[k], reverse=True)
        while excess > 0:
            for key in ordered:
                if quotas[key] > 1:
                    quotas[key] -= 1
                    excess -= 1
                    if excess == 0:
                        break
    return quotas


def main() -> None:
    seed = 42
    random.seed(seed)

    src = Path("data/triples/merged/merged_triples.json")
    triples = load_triples(src)
    total_avail = len(triples)
    if total_avail == 0:
        raise SystemExit("No triples found in source; aborting.")

    strata = build_strata(triples)
    quotas = compute_quotas(strata, total=60)

    selected: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for key, items in strata.items():
        q = min(quotas.get(key, 0), len(items))
        if q <= 0:
            continue
        batch = random.sample(items, k=q)
        selected.extend(batch)
        for b in batch:
            seen_ids.add(id(b))

    # Fill to 60 if short
    if len(selected) < 60:
        pool = [t for t in triples if id(t) not in seen_ids]
        if pool:
            take = min(60 - len(selected), len(pool))
            selected.extend(random.sample(pool, take))

    selected = selected[:60]

    triples_out: list[dict[str, Any]] = []
    per_type: dict[str, int] = {}
    per_rel: dict[str, int] = {}
    for idx, t in enumerate(selected, start=1):
        head_type = t.get("head_type", "unknown")
        relation = t.get("relation", "unknown")
        triples_out.append(
            {
                "id": idx,
                "head": t.get("head", ""),
                "head_type": head_type,
                "relation": relation,
                "tail": t.get("tail", ""),
                "tail_type": t.get("tail_type", ""),
                "evidence": first_evidence(t),
                "correct": None,
                "corrected_head": None,
                "corrected_head_type": None,
                "corrected_relation": None,
                "corrected_tail": None,
                "corrected_tail_type": None,
                "corrected_evidence": None,
                "notes": None,
            }
        )
        per_type[head_type] = per_type.get(head_type, 0) + 1
        per_rel[relation] = per_rel.get(relation, 0) + 1

    output = {
        "meta": {
            "source": str(src),
            "total_source_triples": total_avail,
            "sampled_count": len(triples_out),
            "seed": seed,
            "strategy": "stratified_random",
            "annotation_instructions": (
                "Set 'correct' to true if head/relation/tail/evidence are all correct. "
                "If false, fill corrected_* fields and optionally add notes."
            ),
        },
        "triples": triples_out,
        "stats": {
            "per_entity_type": dict(sorted(per_type.items())),
            "per_relation": dict(sorted(per_rel.items())),
        },
    }

    out_path = Path("data/eval/gold_triples.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Sampled {len(triples_out)} triples from {total_avail} available.\n")
    print("Per-entity-type distribution:")
    for k in sorted(per_type):
        print(f"  {k}: {per_type[k]}")
    print("\nPer-relation distribution:")
    for r in sorted(per_rel):
        print(f"  {r}: {per_rel[r]}")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
