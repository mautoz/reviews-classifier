from helpers import db_aux

def start():
    try:
        with db_aux.connect_db as conn:
             

if __name__ == "__main__":
    start()