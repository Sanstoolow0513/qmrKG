"""CLI for importing merged triples into Neo4j."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .kg_neo4j import KGNeo4jLoader


def main():
    parser = argparse.ArgumentParser(description="Import KG triples into Neo4j")
    parser.add_argument(
        "--import",
        dest="import_file",
        type=Path,
        help="Path to merged triples JSON file to import",
    )
    parser.add_argument("--uri", type=str, help="Neo4j URI (default: env NEO4J_URI)")
    parser.add_argument("--user", type=str, help="Neo4j user (default: env NEO4J_USER)")
    parser.add_argument(
        "--password", type=str, help="Neo4j password (default: env NEO4J_PASSWORD)"
    )
    parser.add_argument("--clear", action="store_true", help="Clear database before import")
    parser.add_argument("--stats", action="store_true", help="Print database statistics")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.import_file and not args.stats:
        parser.print_help()
        sys.exit(1)

    with KGNeo4jLoader(uri=args.uri, user=args.user, password=args.password) as loader:
        if args.import_file:
            if not args.import_file.exists():
                print(f"File not found: {args.import_file}", file=sys.stderr)
                sys.exit(1)
            result = loader.import_from_file(args.import_file, clear=args.clear)
            print(
                f"Imported: {result['entities_created']} entities, "
                f"{result['relations_created']} relations"
            )

        if args.stats:
            stats = loader.get_stats()
            print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
