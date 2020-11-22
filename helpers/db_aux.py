import psycopg2 as db
from contextlib import contextmanager

@contextmanager
def connect_db(db_credentials):
    conn = db.connect(**db_credentials)
    print("Conectando ao BD!")
    yield conn
    conn.close()

def insert_db(conn, table, review):
    try:
        cur = conn.cursor()
        query = f"INSERT INTO {table} ( \
                scraper_date, \
                review_source, \
                review_app, \
                review_language, \
                review_raw \
            ) VALUES (%s, %s, %s, %s, %s)"
        values = (
            review["scraper_date"],
            review["source"],
            review["app_name"],
            review["language"],
            review["review_content"]
        )
        cur.execute(query, values)
        print("+ Inserting...")
        conn.commit()
    except Exception as err:
        print(err)
        print("- Something get wrong! The file wasn't inserted!")
        raise


def search_file(conn, table, id):
    cur = conn.cursor()
    cur.execute(f"SELECT review_app, review_raw FROM {table} WHERE id={id}")
    return cur.fetchone()