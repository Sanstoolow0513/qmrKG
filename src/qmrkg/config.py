"""Project configuration loader.

Runtime behavior is intentionally closed over this module: Python defaults live in
``DEFAULT_RUN_CONFIG`` and ``config.yaml`` may override them through the top-level
``run`` section. CLI entry points should only select which config file to read.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RUN_CONFIG: dict[str, dict[str, Any]] = {
    "qmr": {
        "from_stage": None,
        "to_stage": None,
        "no_neo4j": False,
    },
    "pdf_to_png": {
        "input_file": None,
        "pdf_dir": "data/pdf",
        "image_dir": "data/png",
        "dpi": 200,
        "recursive": False,
        "libreoffice": "libreoffice",
        "ppt_timeout": 300,
    },
    "png_to_text": {
        "input_file": None,
        "output": None,
        "image_dir": "data/png",
        "text_dir": "data/markdown",
        "recursive": True,
        "lang": "ch",
        "gpu": False,
        "force_ocr": False,
    },
    "md_chunk": {
        "input_file": None,
        "output": None,
        "markdown_dir": "data/markdown",
        "chunk_dir": "data/chunks",
        "max_tokens": 4000,
        "recursive": False,
    },
    "kg_md_combine": {
        "markdown_dir": "data/markdown",
        "page_glob": "*_page_*.md",
    },
    "kg_extract": {
        "input": "data/chunks",
        "output_dir": None,
        "mode": "fs",
        "no_skip": False,
        "review": False,
        "strict_evidence": True,
        "keep_dropped": True,
        "extractor_version": "kgextract_v2",
    },
    "kg_merge": {
        "input_dir": "data/triples/raw-fs",
        "output": "data/triples/merged/merged_triples.json",
        "embedding": {
            "enabled": True,
            "task_name": "entity_embed",
            "encode_fields": ["type", "name", "description"],
            "candidate_threshold": 0.78,
            "similarity_threshold": 0.85,
            "direct_merge_without_recheck": False,
            "bucket_by_type": True,
            "batch_size": 1024,
            "cache_path": "data/triples/merged/.embed_cache.json",
            "llm_recheck": {
                "enabled": True,
                "task_name": "entity_merge_review",
                "max_pairs": 200,
                "context_triples_per_entity": 8,
                "require_supporting_evidence": True,
                "allow_merge_with_truncated_context": False,
                "max_cluster_size": 4,
                "require_complete_pairwise_cluster": True,
            },
        },
    },
    "kg_neo4j": {
        "import_file": "data/triples/merged/merged_triples.json",
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "clear": False,
        "stats": False,
    },
    "kg_eval": {
        "pred": "data/triples/merged/merged_triples.json",
        "gold": None,
        "output_json": None,
        "output_md": None,
        "top_errors": 10,
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "run": DEFAULT_RUN_CONFIG,
}

_LEGACY_RUN_KEYS: dict[str, dict[str, str]] = {
    "pdf_to_png": {"pdf": "input_file"},
    "png_to_text": {"image": "input_file"},
    "md_chunk": {"markdown": "input_file"},
    "kg_neo4j": {"import": "import_file"},
}


def _discover_config_paths(config_path: Path | None) -> list[Path]:
    if config_path is not None:
        return [Path(config_path).expanduser().resolve()]

    repo_root = Path(__file__).resolve().parents[2]
    return [
        Path.cwd() / "config.yaml",
        Path.cwd() / "config.yml",
        repo_root / "config.yaml",
        repo_root / "config.yml",
    ]


def _deep_merge(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `default` (dict values recurse)."""
    result: dict[str, Any] = deepcopy(default)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _merge_sections(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(default)
    for section, values in override.items():
        if section not in merged:
            continue
        if not isinstance(values, dict):
            logger.warning("Ignoring non-mapping run.%s config: %r", section, values)
            continue
        merged[section] = _deep_merge(merged[section], values)
    return merged


def _load_yaml_config(config_path: Path | None = None) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not installed, using default project config")
        return {}

    for path in _discover_config_paths(config_path):
        if not path.exists():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            logger.debug("Loaded config from %s", path)
            if not isinstance(data, dict):
                logger.warning("Ignoring non-mapping config file %s: %r", path, data)
                return {}
            return data
        except Exception as exc:
            logger.warning("Failed to load config from %s: %s", path, exc)
    return {}


def _normalize_run_section(run_section: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(run_section)
    for stage, mapping in _LEGACY_RUN_KEYS.items():
        stage_cfg = normalized.get(stage)
        if not isinstance(stage_cfg, dict):
            continue
        for old_key, new_key in mapping.items():
            if old_key in stage_cfg and new_key not in stage_cfg:
                stage_cfg[new_key] = stage_cfg[old_key]
    return normalized


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load project config with Python defaults and YAML overrides."""
    config_data = _load_yaml_config(config_path)
    run_section = config_data.get("run", {})
    if run_section in (None, ""):
        run_section = {}
    if not isinstance(run_section, dict):
        logger.warning("Ignoring non-mapping 'run' config: %r", run_section)
        run_section = {}

    merged = deepcopy(config_data)
    merged["run"] = _merge_sections(DEFAULT_RUN_CONFIG, _normalize_run_section(run_section))
    return merged


def load_run_config(config_path: Path | None = None) -> dict[str, dict[str, Any]]:
    return load_config(config_path)["run"]


def optional_path(value: Any) -> Path | None:
    """Return a Path for configured values; treat null/empty as disabled."""
    if value in (None, ""):
        return None
    return Path(str(value))
