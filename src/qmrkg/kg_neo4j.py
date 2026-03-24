"""Load merged triples into Neo4j graph database."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from tqdm import tqdm

from .kg_schema import ENTITY_TYPE_LABELS, RELATION_TYPE_LABELS

logger = logging.getLogger(__name__)

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"


def _read_neo4j_env() -> tuple[str, str, str]:
    """Read Neo4j connection settings from environment variables."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    uri = os.getenv("NEO4J_URI", DEFAULT_URI)
    user = os.getenv("NEO4J_USER", DEFAULT_USER)
    password = os.getenv("NEO4J_PASSWORD", "")
    if not password:
        raise ValueError("NEO4J_PASSWORD environment variable is required")
    return uri, user, password


class KGNeo4jLoader:
    """Import merged triples into Neo4j."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise ImportError("neo4j not installed. Run: pip install neo4j") from exc

        env_uri, env_user, env_password = _read_neo4j_env()
        self._uri = uri or env_uri
        self._user = user or env_user
        self._password = password or env_password
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def clear_database(self) -> None:
        """Delete all nodes and relationships."""
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared all data from Neo4j")

    def import_from_file(self, merged_path: Path, clear: bool = False) -> dict:
        """Import merged triples JSON file into Neo4j.

        Returns:
            Dict with counts of created entities and relations.
        """
        data = json.loads(Path(merged_path).read_text(encoding="utf-8"))

        if clear:
            self.clear_database()

        entities_created = self._create_entities(data.get("entities", []))
        relations_created = self._create_relations(data.get("triples", []))

        logger.info("Imported %d entities, %d relations", entities_created, relations_created)
        return {"entities_created": entities_created, "relations_created": relations_created}

    def _create_entities(self, entities: list[dict]) -> int:
        count = 0
        with self._driver.session() as session:
            for entity in tqdm(
                entities,
                desc="kgneo4j entities",
                unit="node",
                total=len(entities),
                dynamic_ncols=True,
            ):
                label = ENTITY_TYPE_LABELS.get(entity.get("type", ""))
                if not label:
                    continue
                session.run(
                    f"MERGE (n:{label} {{name: $name}}) "
                    "SET n.description = $description, n.frequency = $frequency",
                    name=entity["name"],
                    description=entity.get("description", ""),
                    frequency=entity.get("frequency", 1),
                )
                count += 1
        return count

    def _create_relations(self, triples: list[dict]) -> int:
        count = 0
        with self._driver.session() as session:
            for triple in tqdm(
                triples,
                desc="kgneo4j relations",
                unit="rel",
                total=len(triples),
                dynamic_ncols=True,
            ):
                head_label = ENTITY_TYPE_LABELS.get(triple.get("head_type", ""))
                tail_label = ENTITY_TYPE_LABELS.get(triple.get("tail_type", ""))
                rel_type = RELATION_TYPE_LABELS.get(triple.get("relation", ""))
                if not head_label or not tail_label or not rel_type:
                    continue
                evidences_json = json.dumps(
                    triple.get("evidences", []), ensure_ascii=False
                )
                session.run(
                    f"MATCH (a:{head_label} {{name: $head}}) "
                    f"MATCH (b:{tail_label} {{name: $tail}}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    "SET r.frequency = $frequency, r.evidences = $evidences",
                    head=triple["head"],
                    tail=triple["tail"],
                    frequency=triple.get("frequency", 1),
                    evidences=evidences_json,
                )
                count += 1
        return count

    def get_stats(self) -> dict:
        """Return node and relationship counts from Neo4j."""
        with self._driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        return {"nodes": node_count, "relationships": rel_count}
