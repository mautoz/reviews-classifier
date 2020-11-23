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
# Header do CSV: reviewId, userName, userImage, content, score, thumbsUpCount
# reviewCreatedVersion, at, replyContent, repliedAt, app_Id
# Para poupar memória, é selecionado somente as colunas necessárias
reviews_columns = ['content', 'app_Id']
reviews = pd.read_csv('csv/Google_Play_Store_Apps_Reviews/google_play_store_apps_reviews.csv', usecols=reviews_columns, nrows=6)

# Selecionando as colunas que interessam no CSV com o nome dos APPs e abrindo o CSV
appId_columns = ['appId', 'title']
apps_names = pd.read_csv('csv/Google_Play_Store_Apps_Reviews/apps.csv', usecols=appId_columns)
apps_names.dropna(inplace = True) 
names = apps_names["appId"].str.find('ginlemon.smartlauncher.extratool')

# Localiza o nome formatado do AppId 
def find_app_name(appId):
    name_line = apps_names.loc[apps_names["appId"] == appId]
    return name_line.applymap(str).iloc[0,1]


print("Starting Google Apps Reader")


with db_aux.connect_db(db_credentials) as conn:
    for linha in range(len(reviews)):
        review = {}
        review["scraper_date"] = datetime.datetime.now()
        review["source"] = "Google"
        review["app_name"] = str(find_app_name(reviews["app_Id"][linha])).strip()
        review["language"] = "en"
        review["review_content"] = reviews["content"][linha]
        print(review)
        db_aux.insert_db(conn, "reviews_data", review)
        # print(f'{reviews["content"][linha]} - {find_app_name(reviews["app_Id"][linha])}')

    example = db_aux.search_file(conn, "reviews_data", 8)
    print(f'{example[0]} - {example[1]}')
    

print(len(reviews))
