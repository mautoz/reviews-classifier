from datetime import date
import re
import pandas as pd
import os
import datetime
from helpers import format_aux, db_aux

db_credentials = {
    'host' : os.getenv('POSTGRES_HOST'),
    'dbname' : os.getenv('POSTGRES_DATABASE'),
    'user' : os.getenv('POSTGRES_USER'),
    'password' : os.getenv('POSTGRES_PASSWORD'),
    'port' : os.getenv('POSTGRES_PORT') 
}


# Google play store apps reviews 
# Colunas no header do CSV que realmente importam: App, Translated_Review 
reviews_columns = ['App', 'Translated_Review']
reviews = pd.read_csv('csv/Google_Play_Store_Extended/extended_googleplaystore_user_reviews.csv', usecols=reviews_columns)


print("Starting Google Apps Extended Reader")


with db_aux.connect_db(db_credentials) as conn:
    for linha in range(len(reviews)):
        review = {}
        review["scraper_date"] = datetime.datetime.now()
        review["source"] = "Google"
        review["app_name"] = str(reviews["App"][linha]).strip()
        review["language"] = "en"
        review["review_content"] = reviews["Translated_Review"][linha]
        print(review)
        db_aux.insert_db(conn, "reviews_data", review)


# for linha in range(len(reviews)):
#     print(f'{reviews["App"][linha]} - {reviews["Translated_Review"][linha]}')

