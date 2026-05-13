"""Relaxed evaluation of raw per-chunk extraction against gold.

Aggregates raw triples across all chunks that gold covers, type-enriches
them via the per-chunk entity list, then applies the same relaxed
matching tiers as scripts/eval_relaxed.py:

  Strict / Name-Norm / Substring / Head+Relation / Relation-level

Usage:
    python scripts/eval_relaxed_raw.py \
        --raw-dir data/triples/raw-fs-recheck \
        --gold data/eval/gold_triples.json \
        --name fs-recheck
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Reuse helpers from eval_relaxed
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_relaxed import (  # noqa: E402
    norm_name,
    triple_keys_strict,
    triple_keys_norm,
    triple_keys_relation,
    triple_substring_match,
    triple_head_relation_match,
    prf,
    fmt_pct,
    _row,
)


def collect_raw(raw_dir: Path, gold_path: Path) -> dict:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for t in gold.get("triples", []) or []:
        sf = str(t.get("source_file", "")).strip()
        ci = int(t.get("chunk_index", 0))
        if sf:
            grouped[(sf, ci)].append(t)

    pred_triples_payload = {"triples": []}
    chunks_matched = 0
    chunks_missing = 0

    for (sf, ci) in grouped.keys():
        stem = Path(sf).stem
        rp = raw_dir / f"{stem}_chunk_{ci:04d}.json"
        if not rp.exists():
            chunks_missing += 1
            continue
        chunks_matched += 1
        raw = json.loads(rp.read_text(encoding="utf-8"))
        type_map = {
            (e.get("name") or "").strip(): (e.get("type") or "").strip()
            for e in raw.get("entities", []) or []
            if e.get("name")
        }
        for t in raw.get("triples", []) or []:
            h = (t.get("head") or "").strip()
            r = (t.get("relation") or "").strip()
            tl = (t.get("tail") or "").strip()
            if not (h and r and tl):
                continue
            ht = type_map.get(h, "")
            tt = type_map.get(tl, "")
            pred_triples_payload["triples"].append({
                "head": h, "head_type": ht, "relation": r,
                "tail": tl, "tail_type": tt,
                "evidence": t.get("evidence", ""),
            })

    return {
        "pred": pred_triples_payload,
        "gold": gold,
        "chunks_matched": chunks_matched,
        "chunks_missing": chunks_missing,
        "gold_chunks": len(grouped),
    }


def evaluate(raw_dir: Path, gold_path: Path) -> dict:
    bundle = collect_raw(raw_dir, gold_path)
    pred = bundle["pred"]
    gold = bundle["gold"]

    pred_t_strict = triple_keys_strict(pred)
    gold_t_strict = triple_keys_strict(gold)
    t_strict = (
        len(pred_t_strict & gold_t_strict),
        len(pred_t_strict - gold_t_strict),
        len(gold_t_strict - pred_t_strict),
    )

    pred_t_norm = triple_keys_norm(pred)
    gold_t_norm = triple_keys_norm(gold)
    t_norm = (
        len(pred_t_norm & gold_t_norm),
        len(pred_t_norm - gold_t_norm),
        len(gold_t_norm - pred_t_norm),
    )

    t_substr = triple_substring_match(pred, gold)
    t_hrel = triple_head_relation_match(pred, gold)

    pred_rel = triple_keys_relation(pred)
    gold_rel = triple_keys_relation(gold)
    rel = (
        len(pred_rel & gold_rel),
        len(pred_rel - gold_rel),
        len(gold_rel - pred_rel),
    )

    # Entity-level strict/norm
    pred_ents: set[tuple[str, str]] = set()
    gold_ents: set[tuple[str, str]] = set()
    for t in pred["triples"]:
        if t["head_type"]:
            pred_ents.add((t["head"], t["head_type"]))
        if t["tail_type"]:
            pred_ents.add((t["tail"], t["tail_type"]))
    for t in gold["triples"]:
        gold_ents.add((t["head"], t["head_type"]))
        gold_ents.add((t["tail"], t["tail_type"]))

    e_strict = (
        len(pred_ents & gold_ents),
        len(pred_ents - gold_ents),
        len(gold_ents - pred_ents),
    )

    pred_ents_norm = {(norm_name(n), t) for n, t in pred_ents}
    gold_ents_norm = {(norm_name(n), t) for n, t in gold_ents}
    e_norm = (
        len(pred_ents_norm & gold_ents_norm),
        len(pred_ents_norm - gold_ents_norm),
        len(gold_ents_norm - pred_ents_norm),
    )

    return {
        "raw_dir": str(raw_dir),
        "gold_path": str(gold_path),
        "chunks_matched": bundle["chunks_matched"],
        "chunks_missing": bundle["chunks_missing"],
        "gold_chunks": bundle["gold_chunks"],
        "pred_triples": len(pred["triples"]),
        "gold_triples": len(gold["triples"]),
        "entity": {
            "strict": _row(e_strict),
            "name_norm": _row(e_norm),
        },
        "triple": {
            "strict": _row(t_strict),
            "name_norm": _row(t_norm),
            "substring": _row(t_substr),
            "head_relation": _row(t_hrel),
        },
        "relation_level": _row(rel),
    }


def render_console(name: str, rep: dict) -> str:
    lines = [
        f"## {name}",
        "",
        f"- chunks: matched {rep['chunks_matched']}, missing {rep['chunks_missing']}, "
        f"gold {rep['gold_chunks']}",
        f"- pred triples: {rep['pred_triples']}, gold triples: {rep['gold_triples']}",
        "",
        "| Scope | Tier | TP | FP | FN | P | R | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in ("strict", "name_norm"):
        e = rep["entity"][tier]
        lines.append(
            f"| Entity | {tier} | {e['tp']} | {e['fp']} | {e['fn']} | "
            f"{fmt_pct(e['precision'])} | {fmt_pct(e['recall'])} | {fmt_pct(e['f1'])} |"
        )
    for tier in ("strict", "name_norm", "substring", "head_relation"):
        t = rep["triple"][tier]
        lines.append(
            f"| Triple | {tier} | {t['tp']} | {t['fp']} | {t['fn']} | "
            f"{fmt_pct(t['precision'])} | {fmt_pct(t['recall'])} | {fmt_pct(t['f1'])} |"
        )
    rl = rep["relation_level"]
    lines.append(
        f"| Relation | type-only | {rl['tp']} | {rl['fp']} | {rl['fn']} | "
        f"{fmt_pct(rl['precision'])} | {fmt_pct(rl['recall'])} | {fmt_pct(rl['f1'])} |"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    rep = evaluate(Path(args.raw_dir), Path(args.gold))
    print(render_console(args.name or args.raw_dir, rep))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
