"""Relaxed evaluation of merged KG against gold.

Reports four tiers of P/R/F1 (entity- and triple-level) plus a
relation-level (type-only) reference column:

- Strict          : exact (name, type) and exact (h, ht, r, t, tt)
- Name-Norm       : entity name normalized (case-fold, whitespace-fold,
                    punctuation strip, parenthetical strip)
- +Alias          : Name-Norm extended with description-as-alias (the
                    `description` field on merged entities is treated
                    as an additional name key, enabling Chinese/English
                    cross-language matching)
- Relation-level  : (head_type, relation, tail_type) only (ignores names)

Usage:
    python scripts/eval_relaxed.py \
        --pred data/triples/merged-eval-gold500/fs-recheck/merged_triples.json \
        --gold data/eval/gold_triples.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable

PUNCT_RE = re.compile(r"[\s　\-_/\\.,;:!?\"'`~@#$%^&*+=<>|()\[\]{}—–·、，。；：！？「」『』《》【】（）]+")
PAREN_RE = re.compile(r"[\(（][^\)）]*[\)）]")


def norm_name(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = PAREN_RE.sub("", s)
    s = s.lower()
    s = PUNCT_RE.sub("", s)
    return s.strip()


def aliases_for(name: str, description: str = "") -> set[str]:
    out = set()
    n = norm_name(name)
    if n:
        out.add(n)
    if description:
        for piece in re.split(r"[,，;；/、]| or | and |\s+又称\s+|\s+又叫\s+", description):
            p = norm_name(piece)
            if p and 1 < len(p) < 60:
                out.add(p)
    return out


def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def build_pred_entity_index(pred: dict) -> tuple[set[tuple[str, str]], dict[tuple[str, str], set[str]]]:
    """Returns:
    - strict entity set  : {(name, type)}
    - alias index        : {(alias_norm, type) -> {original_name, ...}} for relaxed lookup
    """
    strict: set[tuple[str, str]] = set()
    alias_idx: dict[tuple[str, str], set[str]] = defaultdict(set)
    for e in pred.get("entities", []):
        name = (e.get("name") or "").strip()
        etype = (e.get("type") or "").strip()
        desc = (e.get("description") or "").strip()
        if not name or not etype:
            continue
        strict.add((name, etype))
        for a in aliases_for(name, desc):
            alias_idx[(a, etype)].add(name)
    return strict, alias_idx


def build_gold_entity_index(gold: dict) -> tuple[set[tuple[str, str]], dict[tuple[str, str], set[str]]]:
    strict: set[tuple[str, str]] = set()
    alias_idx: dict[tuple[str, str], set[str]] = defaultdict(set)
    for e in gold.get("entities", []) or []:
        name = (e.get("name") or "").strip()
        etype = (e.get("type") or "").strip()
        if not name or not etype:
            continue
        strict.add((name, etype))
        for a in aliases_for(name, ""):
            alias_idx[(a, etype)].add(name)
    if not strict:
        for t in gold.get("triples", []) or []:
            for role in ("head", "tail"):
                name = (t.get(role) or "").strip()
                etype = (t.get(f"{role}_type") or "").strip()
                if not name or not etype:
                    continue
                strict.add((name, etype))
                for a in aliases_for(name, ""):
                    alias_idx[(a, etype)].add(name)
    return strict, alias_idx


def relaxed_set_match(
    pred: set[tuple[str, str]],
    gold: set[tuple[str, str]],
    pred_idx: dict[tuple[str, str], set[str]],
    gold_idx: dict[tuple[str, str], set[str]],
    use_alias: bool,
) -> tuple[int, int, int]:
    """Returns (tp, fp, fn) under relaxed entity matching.

    A predicted entity is a TP if its normalized-name (or any alias if
    use_alias=True) intersects with any gold entity of the same type.
    Symmetrically a gold entity is recalled if any prediction maps to it.
    """
    if use_alias:
        p_norm: dict[tuple[str, str], set[str]] = pred_idx
        g_norm: dict[tuple[str, str], set[str]] = gold_idx
    else:
        p_norm = defaultdict(set)
        for (n, t) in pred:
            p_norm[(norm_name(n), t)].add(n)
        g_norm = defaultdict(set)
        for (n, t) in gold:
            g_norm[(norm_name(n), t)].add(n)

    pred_keys = set(p_norm.keys())
    gold_keys = set(g_norm.keys())
    tp_keys = pred_keys & gold_keys
    fp_keys = pred_keys - gold_keys
    fn_keys = gold_keys - pred_keys
    return len(tp_keys), len(fp_keys), len(fn_keys)


def triple_keys_strict(payload: dict) -> set[tuple[str, str, str, str, str]]:
    out = set()
    for t in payload.get("triples", []) or []:
        h = (t.get("head") or "").strip()
        ht = (t.get("head_type") or "").strip()
        r = (t.get("relation") or "").strip()
        tl = (t.get("tail") or "").strip()
        tt = (t.get("tail_type") or "").strip()
        if h and ht and r and tl and tt:
            out.add((h, ht, r, tl, tt))
    return out


def triple_keys_norm(
    payload: dict,
    alias_index: dict[tuple[str, str], set[str]] | None = None,
) -> set[tuple[str, str, str, str, str]]:
    """Project each triple to a canonical key.

    If alias_index is provided, replace each endpoint name by the smallest
    alias key for that (norm_name, type) bucket so two payloads using
    different surface forms collapse onto the same key.
    """
    out: set[tuple[str, str, str, str, str]] = set()
    for t in payload.get("triples", []) or []:
        h = (t.get("head") or "").strip()
        ht = (t.get("head_type") or "").strip()
        r = (t.get("relation") or "").strip()
        tl = (t.get("tail") or "").strip()
        tt = (t.get("tail_type") or "").strip()
        if not (h and ht and r and tl and tt):
            continue
        h_key = norm_name(h)
        t_key = norm_name(tl)
        out.add((h_key, ht, r, t_key, tt))
    return out


def triple_keys_relation(payload: dict) -> set[tuple[str, str, str]]:
    out = set()
    for t in payload.get("triples", []) or []:
        ht = (t.get("head_type") or "").strip()
        r = (t.get("relation") or "").strip()
        tt = (t.get("tail_type") or "").strip()
        if ht and r and tt:
            out.add((ht, r, tt))
    return out


def _name_substring_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) < 2 or len(b) < 2:
        return False
    return (a in b) or (b in a)


def triple_substring_match(
    pred: dict,
    gold: dict,
) -> tuple[int, int, int]:
    """Triple match where each endpoint name matches via substring containment
    (after norm_name) and types match exactly. This captures cases like
    pred="TCP拥塞控制" vs gold="拥塞控制" or pred="传输控制协议(TCP)" vs gold="TCP".
    """
    pred_list = []
    for t in pred.get("triples", []) or []:
        h = norm_name(t.get("head", ""))
        ht = (t.get("head_type") or "").strip()
        r = (t.get("relation") or "").strip()
        tl = norm_name(t.get("tail", ""))
        tt = (t.get("tail_type") or "").strip()
        if h and ht and r and tl and tt:
            pred_list.append((h, ht, r, tl, tt))

    gold_list = []
    for t in gold.get("triples", []) or []:
        h = norm_name(t.get("head", ""))
        ht = (t.get("head_type") or "").strip()
        r = (t.get("relation") or "").strip()
        tl = norm_name(t.get("tail", ""))
        tt = (t.get("tail_type") or "").strip()
        if h and ht and r and tl and tt:
            gold_list.append((h, ht, r, tl, tt))

    matched_pred = set()
    matched_gold = set()
    for i, (gh, ght, gr, gtl, gtt) in enumerate(gold_list):
        for j, (ph, pht, pr, ptl, ptt) in enumerate(pred_list):
            if pr != gr or pht != ght or ptt != gtt:
                continue
            if _name_substring_match(gh, ph) and _name_substring_match(gtl, ptl):
                matched_gold.add(i)
                matched_pred.add(j)
    tp = len(matched_gold)
    fn = len(gold_list) - tp
    fp = len(pred_list) - len(matched_pred)
    return tp, fp, fn


def triple_head_relation_match(pred: dict, gold: dict) -> tuple[int, int, int]:
    """Looser: a gold triple is recalled if pred contains any triple with the
    same (head_norm-or-substring, head_type, relation, tail_type). Tail name
    is ignored. Captures cases where the model picks a different but correct
    tail entity for the same relation."""
    def project(payload):
        return [
            (norm_name(t.get("head", "")), (t.get("head_type") or "").strip(),
             (t.get("relation") or "").strip(), (t.get("tail_type") or "").strip())
            for t in payload.get("triples", []) or []
            if t.get("head") and t.get("relation")
        ]

    pred_list = project(pred)
    gold_list = project(gold)
    matched_pred = set()
    matched_gold = set()
    for i, (gh, ght, gr, gtt) in enumerate(gold_list):
        for j, (ph, pht, pr, ptt) in enumerate(pred_list):
            if pr != gr or pht != ght or ptt != gtt:
                continue
            if _name_substring_match(gh, ph):
                matched_gold.add(i)
                matched_pred.add(j)
    tp = len(matched_gold)
    fn = len(gold_list) - tp
    fp = len(pred_list) - len(matched_pred)
    return tp, fp, fn


def triple_keys_alias_match(
    pred: dict,
    gold: dict,
    pred_alias_idx: dict[tuple[str, str], set[str]],
    gold_alias_idx: dict[tuple[str, str], set[str]],
) -> tuple[int, int, int]:
    """Count tp/fp/fn at triple level using alias-aware endpoint matching.

    A predicted triple matches a gold triple if their relations are equal
    AND their (head_type, head_alias) buckets share at least one alias
    AND the same holds for tails. We collapse each predicted triple to a
    canonical key by replacing endpoint names with one element of the
    overlap with gold (if any), else with the predicted norm name.
    """
    # Build alias -> canonical key per (type) bucket using gold side as anchor.
    def build_canonical(p_idx, g_idx):
        canon: dict[tuple[str, str], str] = {}
        for k, names in g_idx.items():
            canon[k] = sorted(names)[0]
        return canon

    g_canon = build_canonical(pred_alias_idx, gold_alias_idx)

    def key_for(name: str, etype: str, alias_idx: dict[tuple[str, str], set[str]]):
        nn = norm_name(name)
        if (nn, etype) in g_canon:
            return g_canon[(nn, etype)]
        # try description aliases of this name in pred index
        for k, names in alias_idx.items():
            if name in names and k in g_canon:
                return g_canon[k]
        return nn

    pred_keys: set[tuple[str, str, str, str, str]] = set()
    for t in pred.get("triples", []) or []:
        h = (t.get("head") or "").strip()
        ht = (t.get("head_type") or "").strip()
        r = (t.get("relation") or "").strip()
        tl = (t.get("tail") or "").strip()
        tt = (t.get("tail_type") or "").strip()
        if not (h and ht and r and tl and tt):
            continue
        pred_keys.add((key_for(h, ht, pred_alias_idx), ht, r, key_for(tl, tt, pred_alias_idx), tt))

    gold_keys: set[tuple[str, str, str, str, str]] = set()
    for t in gold.get("triples", []) or []:
        h = (t.get("head") or "").strip()
        ht = (t.get("head_type") or "").strip()
        r = (t.get("relation") or "").strip()
        tl = (t.get("tail") or "").strip()
        tt = (t.get("tail_type") or "").strip()
        if not (h and ht and r and tl and tt):
            continue
        gold_keys.add((key_for(h, ht, gold_alias_idx), ht, r, key_for(tl, tt, gold_alias_idx), tt))

    tp = len(pred_keys & gold_keys)
    fp = len(pred_keys - gold_keys)
    fn = len(gold_keys - pred_keys)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(pred_path: Path, gold_path: Path) -> dict:
    pred = load_json(pred_path)
    gold = load_json(gold_path)

    pred_strict_e, pred_idx = build_pred_entity_index(pred)
    gold_strict_e, gold_idx = build_gold_entity_index(gold)

    # Entities
    e_strict = (
        len(pred_strict_e & gold_strict_e),
        len(pred_strict_e - gold_strict_e),
        len(gold_strict_e - pred_strict_e),
    )
    e_norm = relaxed_set_match(pred_strict_e, gold_strict_e, pred_idx, gold_idx, use_alias=False)
    e_alias = relaxed_set_match(pred_strict_e, gold_strict_e, pred_idx, gold_idx, use_alias=True)

    # Triples
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

    t_alias = triple_keys_alias_match(pred, gold, pred_idx, gold_idx)
    t_substr = triple_substring_match(pred, gold)
    t_hrel = triple_head_relation_match(pred, gold)

    pred_rel = triple_keys_relation(pred)
    gold_rel = triple_keys_relation(gold)
    rel = (
        len(pred_rel & gold_rel),
        len(pred_rel - gold_rel),
        len(gold_rel - pred_rel),
    )

    return {
        "pred_path": str(pred_path),
        "gold_path": str(gold_path),
        "pred_entities": len(pred_strict_e),
        "gold_entities": len(gold_strict_e),
        "pred_triples": len(pred_t_strict),
        "gold_triples": len(gold_t_strict),
        "entity": {
            "strict": _row(e_strict),
            "name_norm": _row(e_norm),
            "alias": _row(e_alias),
        },
        "triple": {
            "strict": _row(t_strict),
            "name_norm": _row(t_norm),
            "alias": _row(t_alias),
            "substring": _row(t_substr),
            "head_relation": _row(t_hrel),
        },
        "relation_level": _row(rel),
    }


def _row(c: tuple[int, int, int]) -> dict:
    p, r, f = prf(*c)
    return {"tp": c[0], "fp": c[1], "fn": c[2], "precision": p, "recall": r, "f1": f}


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def render_console(name: str, rep: dict) -> str:
    lines = [f"## {name}", ""]
    lines.append(
        f"- pred entities: {rep['pred_entities']}, gold entities: {rep['gold_entities']}"
    )
    lines.append(
        f"- pred triples: {rep['pred_triples']}, gold triples: {rep['gold_triples']}"
    )
    lines.append("")
    lines.append("| Scope | Tier | TP | FP | FN | P | R | F1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for tier in ("strict", "name_norm", "alias"):
        e = rep["entity"][tier]
        lines.append(
            f"| Entity | {tier} | {e['tp']} | {e['fp']} | {e['fn']} | "
            f"{fmt_pct(e['precision'])} | {fmt_pct(e['recall'])} | {fmt_pct(e['f1'])} |"
        )
    for tier in ("strict", "name_norm", "alias", "substring", "head_relation"):
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


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="merged_triples.json path")
    ap.add_argument("--gold", required=True, help="gold_triples.json path")
    ap.add_argument("--name", default=None, help="display name in output")
    ap.add_argument("--json-out", default=None, help="dump full report json here")
    args = ap.parse_args(argv)
    rep = evaluate(Path(args.pred), Path(args.gold))
    print(render_console(args.name or args.pred, rep))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
