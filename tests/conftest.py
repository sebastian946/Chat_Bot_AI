"""Fixtures compartidos por todas las pruebas."""
import psycopg
import pytest
from psycopg.rows import DictRow, dict_row

import db.postgress as p


@pytest.fixture(scope="session")
def database_is_up():
    """Se comprueba una sola vez por ejecución, para no esperar el timeout en cada prueba."""
    try:
        psycopg.connect(p.DATABASE_URL, connect_timeout=3).close()
    except psycopg.OperationalError:
        return False
    return True


@pytest.fixture
def conn(database_is_up):
    """Conexión al Postgres real dentro de una transacción que se revierte al final."""
    if not database_is_up:
        pytest.skip("Postgres no está disponible (levántalo con: docker compose up -d db)")
    connection = psycopg.Connection[DictRow].connect(p.DATABASE_URL, row_factory=dict_row)
    try:
        p.create_tables(connection)
        yield connection
    finally:
        connection.rollback()      # nada de lo que hizo la prueba se guarda
        connection.close()
