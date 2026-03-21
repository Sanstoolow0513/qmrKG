import os
from unittest.mock import patch

import pytest

from qmrkg.kg_neo4j import _read_neo4j_env, DEFAULT_URI, DEFAULT_USER


def test_read_neo4j_env_defaults():
    with patch.dict(os.environ, {"NEO4J_PASSWORD": "test123"}, clear=False):
        uri, user, password = _read_neo4j_env()
        assert uri == DEFAULT_URI
        assert user == DEFAULT_USER
        assert password == "test123"


def test_read_neo4j_env_missing_password():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="NEO4J_PASSWORD"):
            _read_neo4j_env()


def test_read_neo4j_env_custom():
    env = {
        "NEO4J_URI": "bolt://custom:7687",
        "NEO4J_USER": "admin",
        "NEO4J_PASSWORD": "secret",
    }
    with patch.dict(os.environ, env, clear=True):
        uri, user, password = _read_neo4j_env()
        assert uri == "bolt://custom:7687"
        assert user == "admin"
        assert password == "secret"


def test_read_neo4j_env_empty_password():
    with patch.dict(os.environ, {"NEO4J_PASSWORD": ""}, clear=True):
        with pytest.raises(ValueError, match="NEO4J_PASSWORD"):
            _read_neo4j_env()
