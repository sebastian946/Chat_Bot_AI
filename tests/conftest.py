"""Fixtures shared by all the tests."""
import psycopg
import pytest
from psycopg.rows import DictRow, dict_row

import db.postgress as p


@pytest.fixture(scope="session")
def database_is_up():
    """Checked once per run, so an unreachable database does not cost a timeout in every test."""
    try:
        psycopg.connect(p.DATABASE_URL, connect_timeout=3).close()
    except psycopg.OperationalError:
        return False
    return True


@pytest.fixture
def conn(database_is_up):
    """Connection to the real Postgres inside a transaction that is rolled back at the end."""
    if not database_is_up:
        pytest.skip("Postgres is not reachable (start it with: docker compose up -d db)")
    connection = psycopg.Connection[DictRow].connect(p.DATABASE_URL, row_factory=dict_row)
    try:
        p.create_tables(connection)
        yield connection
    finally:
        connection.rollback()      # nothing the test did is saved
        connection.close()
