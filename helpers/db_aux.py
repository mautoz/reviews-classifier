import psycopg2 as db
from contextlib import contextmanager

@contextmanager
def connect_db(db_credentials):
    conn = db.connect(**db_credentials)
    print("Conectando ao BD!")
    yield conn
    conn.close()

def busca(conn):
    print("Terminar função")