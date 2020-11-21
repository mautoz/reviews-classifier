import pandas as pd
from helpers import format_aux

# Google play store apps reviews 
# Colunas no header do CSV que realmente importam: commentapp_id,content
reviews_columns = ['app_id', 'content']
reviews = pd.read_csv('csv/Android_App_Dataset/comment.csv', usecols=reviews_columns, nrows=6)

# Selecionando as colunas que interessam no CSV com o nome dos APPs e abrindo o CSV
appId_columns = ['id', 'app_name']
apps_names = pd.read_csv('csv/Android_App_Dataset/app.csv', usecols=appId_columns)
apps_names.dropna(inplace = True) 

# Localiza o nome formatado do AppId 
def find_app_name(id):
    name_line = apps_names.loc[apps_names["id"] == id]
    return name_line.applymap(str).iloc[0,1]


for linha in range(len(reviews)):
    print(f'{find_app_name(reviews["app_id"][linha])} - {reviews["content"][linha]}')

