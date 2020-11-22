import pandas as pd
from helpers import format_aux


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
reviews = pd.read_csv('csv/Google_Play_Store_Extended/extended_googleplaystore_user_reviews.csv', usecols=reviews_columns, nrows=6)

for linha in range(len(reviews)):
    print(f'{reviews["App"][linha]} - {reviews["Translated_Review"][linha]}')

