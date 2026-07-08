"""Fixtures for DB-touching (@pytest.mark.integration) tests.

Integration tests run against a dedicated `nasa_rag_test` database on the same
Postgres server as DATABASE_URL, created fresh at session start and dropped at
session end. If that server isn't reachable (e.g. `make db-up` wasn't run),
the fixture skips every test that depends on it rather than failing the suite.
"""

from __future__ import annotations

import psycopg
import pytest

from app.config import settings
from app.ingestion.db import get_connection, init_schema

TEST_DB_NAME = "nasa_rag_test"


def _maintenance_dsn() -> str:
    return settings.DATABASE_URL.rsplit("/", 1)[0] + "/postgres"


def _test_db_dsn() -> str:
    return settings.DATABASE_URL.rsplit("/", 1)[0] + f"/{TEST_DB_NAME}"


def _server_reachable() -> bool:
    try:
        with psycopg.connect(_maintenance_dsn(), connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except psycopg.OperationalError:
        return False


@pytest.fixture(scope="session")
def test_database_url():
    if not _server_reachable():
        pytest.skip("Postgres is not reachable (is `make db-up` running?); skipping integration tests.")

    admin_conn = psycopg.connect(_maintenance_dsn(), autocommit=True)
    admin_conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
    admin_conn.execute(f"CREATE DATABASE {TEST_DB_NAME}")
    admin_conn.close()

    yield _test_db_dsn()

    admin_conn = psycopg.connect(_maintenance_dsn(), autocommit=True)
    admin_conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
    admin_conn.close()


@pytest.fixture
def db_conn(test_database_url):
    conn = get_connection(test_database_url)
    init_schema(conn)
    yield conn
    conn.close()
