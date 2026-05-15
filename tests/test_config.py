"""Tests for runtime CLI config loading."""

from __future__ import annotations

from pathlib import Path

from qmrkg.config import load_run_config


def test_default_kg_merge_uses_embedding_rules_llm_recheck(tmp_path: Path) -> None:
    run = load_run_config(tmp_path / "missing.yaml")
    emb = run["kg_merge"]["embedding"]
    assert emb["enabled"] is True
    assert emb["direct_merge_without_recheck"] is False
    assert emb["llm_recheck"]["enabled"] is True


def test_load_run_config_deep_merges_kg_merge_embedding(tmp_path: Path) -> None:
    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        "run:\n  kg_merge:\n    embedding:\n      candidate_threshold: 0.8\n",
        encoding="utf-8",
    )
    run = load_run_config(cfg)
    emb = run["kg_merge"]["embedding"]
    assert emb["enabled"] is True
    assert emb["candidate_threshold"] == 0.8
    assert emb["cache_path"] == "data/triples/merged/.embed_cache.json"
    assert emb["task_name"] == "entity_embed"
    assert emb["llm_recheck"]["enabled"] is True
