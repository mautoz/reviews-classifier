from helpers import db_aux


def main():
    print("Main")
    try:
        with db_aux.connect_db as conn:
            db_aux.busca(conn)
    
    except Exception as e:
        print(e)
             

if __name__ == "__main__":
    main()