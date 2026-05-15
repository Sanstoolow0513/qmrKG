"""LLM-as-judge evaluation: recall of gold triples on a merged KG.

Pipeline:
  1. Entity alignment per type:
       a. exact normalized-name match  (free)
       b. substring containment match  (free; collects candidates)
       c. character/word-token overlap (free; collects candidates)
       d. LLM judge picks which candidates (if any) refer to the same concept
  2. Triple recall:
       For each gold (gh, gr, gt), look up aligned pred entities of gh and gt.
       If pred has any (ph in aligned(gh), gr, pt in aligned(gt)) -> hit.
       Else, if a pred (ph, r', pt) exists with r' != gr but the same endpoints,
       LLM judges whether r' is semantically equivalent to gr; if yes -> hit.

Output: recall against gold (TP/FN/recall), plus a per-triple log.

Usage:
    uv run python scripts/eval_llm_judge.py \
        --pred data/triples/merged-fs-recheck/merged_triples.json \
        --gold data/eval/gold_triples.json \
        --out data/eval/llm_judge_fs_recheck.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure src/ is importable when running directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmrkg.llm_factory import TextTaskProcessor  # noqa: E402

PUNCT_RE = re.compile(
    r"[\s　\-_/\\.,;:!?\"'`~@#$%^&*+=<>|()\[\]{}—–·、，。；：！？「」『』《》【】（）]+"
)
PAREN_RE = re.compile(r"[\(（][^\)）]*[\)）]")
TOKEN_RE = re.compile(r"[一-鿿]|[A-Za-z0-9]+")


def norm_name(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = PAREN_RE.sub("", s)
    s = s.lower()
    s = PUNCT_RE.sub("", s)
    return s.strip()


def tokens_of(s: str) -> set[str]:
    if not s:
        return set()
    s = unicodedata.normalize("NFKC", s).lower()
    out: set[str] = set()
    for tok in TOKEN_RE.findall(s):
        if "一" <= tok <= "鿿":  # single CJK
            out.add(tok)
        elif len(tok) > 1:
            out.add(tok)
    return out


def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def build_pred_index(pred: dict) -> dict[str, list[dict]]:
    """Group pred entities by type."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for e in pred.get("entities", []) or []:
        t = (e.get("type") or "").strip()
        if t:
            by_type[t].append(e)
    return by_type


def build_pred_triples_by_endpoints(pred: dict) -> dict[tuple[str, str], list[dict]]:
    """Index pred triples by (head_name, tail_name) for fast lookup."""
    idx: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in pred.get("triples", []) or []:
        h = (t.get("head") or "").strip()
        tl = (t.get("tail") or "").strip()
        if h and tl:
            idx[(h, tl)].append(t)
    return idx


def find_entity_candidates(
    gold_entity: dict,
    pred_pool: list[dict],
    *,
    max_candidates: int = 10,
) -> tuple[list[dict], str]:
    """Return (candidates, source_tier).

    source_tier:
      - 'exact_norm'   : single candidate, exact normalized-name match (no LLM needed)
      - 'substring'    : substring containment matched some candidates
      - 'token_overlap': fallback to token Jaccard overlap
      - 'empty'        : no candidates after all tiers
    """
    g_name = (gold_entity.get("name") or "").strip()
    g_norm = norm_name(g_name)

    # Tier A: exact normalized-name match.
    exact: list[dict] = []
    for p in pred_pool:
        if norm_name(p.get("name", "")) == g_norm and g_norm:
            exact.append(p)
    if exact:
        return exact[:max_candidates], "exact_norm"

    # Tier B: substring containment on name or description.
    sub: list[dict] = []
    for p in pred_pool:
        p_name = norm_name(p.get("name", ""))
        p_desc = norm_name(p.get("description", ""))
        if not g_norm:
            continue
        if (
            g_norm in p_name
            or p_name in g_norm
            or g_norm in p_desc
        ):
            sub.append(p)
    if sub:
        return sub[:max_candidates], "substring"

    # Tier C: token overlap (Jaccard >= threshold).
    g_tokens = tokens_of(g_name)
    if not g_tokens:
        return [], "empty"

    scored: list[tuple[float, dict]] = []
    for p in pred_pool:
        p_tokens = tokens_of((p.get("name") or "") + " " + (p.get("description") or ""))
        if not p_tokens:
            continue
        inter = g_tokens & p_tokens
        if not inter:
            continue
        union = g_tokens | p_tokens
        jacc = len(inter) / len(union)
        scored.append((jacc, p))
    scored.sort(key=lambda x: -x[0])
    cands = [p for _, p in scored[:max_candidates]]
    if cands:
        return cands, "token_overlap"
    return [], "empty"


def parse_json_strict(text: str) -> dict | None:
    """Pull the first JSON object from an LLM response."""
    if not text:
        return None
    text = text.strip()
    # strip code fences if any
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"```$", "", text).strip()
    # find first { ... } block by depth tracking
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    return None
    return None


def llm_match_entity(
    judge: TextTaskProcessor,
    gold: dict,
    candidates: list[dict],
    *,
    max_desc_chars: int = 200,
) -> tuple[list[int], str]:
    """Ask LLM which candidates align with gold; returns (matched_indices, raw)."""
    cand_lines = []
    for i, c in enumerate(candidates):
        desc = (c.get("description") or "").strip()
        if len(desc) > max_desc_chars:
            desc = desc[:max_desc_chars] + "…"
        cand_lines.append(
            f"  [{i}] name={c.get('name')!r} description={desc!r} "
            f"freq={c.get('frequency')}"
        )
    user = (
        f"gold: name={gold.get('name')!r} type={gold.get('type')!r}\n"
        f"candidates (type={gold.get('type')!r}):\n" + "\n".join(cand_lines)
    )
    resp = judge.run_text(user)
    payload = parse_json_strict(resp.text or "")
    if not isinstance(payload, dict):
        return [], (resp.text or "")[:200]
    raw_idx = payload.get("matched_indices")
    if not isinstance(raw_idx, list):
        return [], (resp.text or "")[:200]
    out: list[int] = []
    for v in raw_idx:
        if isinstance(v, int) and 0 <= v < len(candidates):
            out.append(v)
    return out, (resp.text or "")[:200]


def llm_match_relation(
    judge: TextTaskProcessor,
    gold_rel: str,
    pred_rel: str,
    head: str,
    tail: str,
) -> tuple[bool, str]:
    user = (
        f"gold: head={head!r} relation={gold_rel!r} tail={tail!r}\n"
        f"candidate: head={head!r} relation={pred_rel!r} tail={tail!r}\n"
        "请仅根据 relation 字段判断两条三元组是否表达相同事实。"
    )
    resp = judge.run_text(user)
    payload = parse_json_strict(resp.text or "")
    if not isinstance(payload, dict):
        return False, (resp.text or "")[:200]
    return bool(payload.get("equivalent")), (resp.text or "")[:200]


def align_entities(
    pred_by_type: dict[str, list[dict]],
    gold_entities: list[dict],
    judge: TextTaskProcessor,
    *,
    log_every: int = 50,
) -> dict[tuple[str, str], list[str]]:
    """Returns: {(gold_name, gold_type) -> [pred_name, ...]}."""
    align: dict[tuple[str, str], list[str]] = {}
    n_total = len(gold_entities)
    n_exact = 0
    n_llm_call = 0
    n_llm_hit = 0
    n_empty = 0

    for i, ge in enumerate(gold_entities, 1):
        gname = (ge.get("name") or "").strip()
        gtype = (ge.get("type") or "").strip()
        if not gname or not gtype:
            continue
        pool = pred_by_type.get(gtype, [])
        cands, tier = find_entity_candidates(ge, pool, max_candidates=10)
        if tier == "exact_norm":
            align[(gname, gtype)] = [c["name"] for c in cands]
            n_exact += 1
        elif tier == "empty":
            align[(gname, gtype)] = []
            n_empty += 1
        else:
            n_llm_call += 1
            try:
                idxs, _raw = llm_match_entity(judge, ge, cands)
            except Exception as exc:
                print(f"  [warn] LLM entity match failed for {gname!r}: {exc}", flush=True)
                idxs = []
            matched_names = [cands[i]["name"] for i in idxs]
            align[(gname, gtype)] = matched_names
            if matched_names:
                n_llm_hit += 1
        if i % log_every == 0:
            print(
                f"  [align] {i}/{n_total} exact={n_exact} llm_calls={n_llm_call} "
                f"llm_hits={n_llm_hit} empty={n_empty}",
                flush=True,
            )
    print(
        f"  [align] DONE total={n_total} exact={n_exact} "
        f"llm_calls={n_llm_call} llm_hits={n_llm_hit} empty={n_empty}",
        flush=True,
    )
    return align


def evaluate_recall(
    pred: dict,
    gold: dict,
    align: dict[tuple[str, str], list[str]],
    judge: TextTaskProcessor,
) -> dict[str, Any]:
    pred_index = build_pred_triples_by_endpoints(pred)

    hits: list[dict] = []
    misses: list[dict] = []
    relation_llm_calls = 0
    relation_llm_hits = 0

    for gt in gold.get("triples", []) or []:
        gh = (gt.get("head") or "").strip()
        ght = (gt.get("head_type") or "").strip()
        gr = (gt.get("relation") or "").strip()
        gtl = (gt.get("tail") or "").strip()
        gtt = (gt.get("tail_type") or "").strip()

        aligned_h = align.get((gh, ght), [])
        aligned_t = align.get((gtl, gtt), [])

        hit_record: dict[str, Any] | None = None
        # Exact-relation hit
        for ph in aligned_h:
            for pt in aligned_t:
                for cand in pred_index.get((ph, pt), []):
                    if cand.get("relation") == gr:
                        hit_record = {
                            "gold": gt,
                            "pred": cand,
                            "match_kind": "exact_relation",
                        }
                        break
                if hit_record:
                    break
            if hit_record:
                break

        if not hit_record:
            # Same endpoints, different relation -> LLM judge
            seen_rels: set[str] = set()
            for ph in aligned_h:
                for pt in aligned_t:
                    for cand in pred_index.get((ph, pt), []):
                        rel = cand.get("relation") or ""
                        if rel in seen_rels:
                            continue
                        seen_rels.add(rel)
                        relation_llm_calls += 1
                        try:
                            equiv, _raw = llm_match_relation(
                                judge, gold_rel=gr, pred_rel=rel, head=ph, tail=pt
                            )
                        except Exception as exc:
                            print(
                                f"  [warn] LLM relation match failed: {exc}", flush=True
                            )
                            equiv = False
                        if equiv:
                            relation_llm_hits += 1
                            hit_record = {
                                "gold": gt,
                                "pred": cand,
                                "match_kind": "relation_judge",
                            }
                            break
                    if hit_record:
                        break
                if hit_record:
                    break

        if hit_record:
            hits.append(hit_record)
        else:
            misses.append({"gold": gt, "aligned_h": aligned_h, "aligned_t": aligned_t})

    total = len(gold.get("triples", []) or [])
    tp = len(hits)
    fn = total - tp
    recall = tp / total if total else 0.0
    return {
        "total_gold_triples": total,
        "tp": tp,
        "fn": fn,
        "recall": recall,
        "relation_llm_calls": relation_llm_calls,
        "relation_llm_hits": relation_llm_hits,
        "hits_breakdown": {
            "exact_relation": sum(1 for h in hits if h["match_kind"] == "exact_relation"),
            "relation_judge": sum(1 for h in hits if h["match_kind"] == "relation_judge"),
        },
        "hits": hits,
        "misses": misses,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--align-only", action="store_true",
                    help="Only run entity alignment, skip triple recall (debugging).")
    ap.add_argument("--align-cache", default=None,
                    help="Optional path to cache/reload entity alignment as JSON.")
    args = ap.parse_args()

    pred = load_json(Path(args.pred))
    gold = load_json(Path(args.gold))

    print(f"[load] pred entities={len(pred.get('entities', []))} "
          f"triples={len(pred.get('triples', []))}", flush=True)
    print(f"[load] gold entities={len(gold.get('entities', []))} "
          f"triples={len(gold.get('triples', []))}", flush=True)

    pred_by_type = build_pred_index(pred)
    print(f"[index] pred entity buckets: " + ", ".join(
        f"{k}={len(v)}" for k, v in pred_by_type.items()), flush=True)

    judge_entity = TextTaskProcessor("eval_entity_match")
    judge_relation = TextTaskProcessor("eval_triple_match")

    align: dict[tuple[str, str], list[str]] = {}
    if args.align_cache and Path(args.align_cache).exists():
        print(f"[align] loading cache from {args.align_cache}", flush=True)
        raw = json.loads(Path(args.align_cache).read_text(encoding="utf-8"))
        for item in raw:
            align[(item["gold_name"], item["gold_type"])] = item["matched_pred_names"]
    else:
        gold_entities = gold.get("entities", []) or []
        align = align_entities(pred_by_type, gold_entities, judge_entity)
        if args.align_cache:
            cache_payload = [
                {"gold_name": k[0], "gold_type": k[1], "matched_pred_names": v}
                for k, v in align.items()
            ]
            Path(args.align_cache).write_text(
                json.dumps(cache_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[align] cache saved to {args.align_cache}", flush=True)

    if args.align_only:
        Path(args.out).write_text(
            json.dumps(
                {"align": [
                    {"gold_name": k[0], "gold_type": k[1], "matched": v}
                    for k, v in align.items()
                ]},
                ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 0

    print("[recall] evaluating triple recall...", flush=True)
    rep = evaluate_recall(pred, gold, align, judge_relation)
    print(
        f"[recall] tp={rep['tp']} fn={rep['fn']} recall={rep['recall']:.4f} "
        f"(exact={rep['hits_breakdown']['exact_relation']}, "
        f"judge={rep['hits_breakdown']['relation_judge']})",
        flush=True,
    )

    out_payload = {
        "pred_path": args.pred,
        "gold_path": args.gold,
        "summary": {
            "total_gold_triples": rep["total_gold_triples"],
            "tp": rep["tp"],
            "fn": rep["fn"],
            "recall": rep["recall"],
            "relation_llm_calls": rep["relation_llm_calls"],
            "relation_llm_hits": rep["relation_llm_hits"],
            "hits_breakdown": rep["hits_breakdown"],
        },
        "align_count": {
            "total": len(align),
            "matched": sum(1 for v in align.values() if v),
            "unmatched": sum(1 for v in align.values() if not v),
        },
        "hits": rep["hits"],
        "misses": rep["misses"],
    }
    Path(args.out).write_text(
        json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[out] saved to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
