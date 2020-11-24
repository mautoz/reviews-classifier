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
# Colunas no header do CSV que realmente importam: commentapp_id,content
reviews_columns = ['app_id', 'content']
reviews = pd.read_csv('csv/Android_App_Dataset/comment.csv', usecols=reviews_columns)

# Selecionando as colunas que interessam no CSV com o nome dos APPs e abrindo o CSV
appId_columns = ['id', 'app_name']
apps_names = pd.read_csv('csv/Android_App_Dataset/app.csv', usecols=appId_columns)
apps_names.dropna(inplace = True) 

# Localiza o nome formatado do AppId 
def find_app_name(id):
    name_line = apps_names.loc[apps_names["id"] == id]
    return name_line.applymap(str).iloc[0,1]

print("Starting Android App Dataset Reader")

with db_aux.connect_db(db_credentials) as conn:
    for linha in range(len(reviews)):
        review = {}
        review["scraper_date"] = datetime.datetime.now()
        review["source"] = "Google"
        review["app_name"] = str(find_app_name(reviews["app_id"][linha])).strip()
        review["language"] = "en"
        review["review_content"] = reviews["content"][linha]
        print(review)
        db_aux.insert_db(conn, "reviews_data", review)


# for linha in range(len(reviews)):
#     print(f'{find_app_name(reviews["app_id"][linha])} - {reviews["content"][linha]}')
