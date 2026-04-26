"""Runtime CLI config loader for non-LLM pipeline stages."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RUN_CONFIG: dict[str, dict[str, Any]] = {
    "pdf_to_png": {
        "pdf_dir": "data/pdf",
        "image_dir": "data/png",
        "dpi": 200,
        "recursive": False,
        "libreoffice": "libreoffice",
        "ppt_timeout": 300,
    },
    "png_to_text": {
        "image_dir": "data/png",
        "text_dir": "data/markdown",
        "recursive": True,
        "lang": "ch",
        "gpu": False,
        "force_ocr": False,
    },
    "md_chunk": {
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
        "output_dir": "data/triples/raw",
        "mode": "fs",
        "no_skip": False,
        "review": True,
        "strict_evidence": True,
        "keep_dropped": True,
        "min_triples": 1,
        "extractor_version": "kgextract_v2",
    },
    "kg_merge": {
        "input_dir": "data/triples/raw",
        "output": "data/triples/merged/merged_triples.json",
    },
    "kg_neo4j": {
        "import": "data/triples/merged/merged_triples.json",
        "clear": False,
        "stats": False,
    },
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


def _merge_sections(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(default)
    for section, values in override.items():
        if section not in merged:
            continue
        if not isinstance(values, dict):
            logger.warning("Ignoring non-mapping run.%s config: %r", section, values)
            continue
        merged[section].update(values)
    return merged


def load_run_config(config_path: Path | None = None) -> dict[str, dict[str, Any]]:
    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not installed, using default run config")
        return deepcopy(DEFAULT_RUN_CONFIG)

    config_data: dict[str, Any] = {}
    for path in _discover_config_paths(config_path):
        if not path.exists():
            continue
        try:
            config_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            logger.debug("Loaded run config from %s", path)
            break
        except Exception as exc:
            logger.warning("Failed to load config from %s: %s", path, exc)

    run_section = config_data.get("run", {})
    if run_section in (None, ""):
        run_section = {}
    if not isinstance(run_section, dict):
        logger.warning("Ignoring non-mapping 'run' config: %r", run_section)
        run_section = {}

    return _merge_sections(DEFAULT_RUN_CONFIG, run_section)
