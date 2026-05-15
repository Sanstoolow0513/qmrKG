"""CLI for importing merged triples into Neo4j."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_run_config
from .kg_neo4j import KGNeo4jLoader


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import KG triples into Neo4j")
    parser.add_argument(
        "--config",
        type=Path,
        help="config.yaml path; all stage settings are read from run.kg_neo4j",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_cfg = load_run_config(args.config)["kg_neo4j"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    import_file = Path(str(run_cfg["import_file"])) if run_cfg.get("import_file") else None
    stats = bool(run_cfg["stats"])
    clear = bool(run_cfg["clear"])
    uri = str(run_cfg["uri"]) if run_cfg.get("uri") else None
    user = str(run_cfg["user"]) if run_cfg.get("user") else None

    if not import_file and not stats:
        parser.print_help()
        return 1

    with KGNeo4jLoader(uri=uri, user=user) as loader:
        if import_file:
            if not import_file.exists():
                print(f"File not found: {import_file}", file=sys.stderr)
                return 1
            result = loader.import_from_file(import_file, clear=clear)
            print(
                f"Imported: {result['entities_created']} entities, "
                f"{result['relations_created']} relations"
            )

        if stats:
            db_stats = loader.get_stats()
            print(json.dumps(db_stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
