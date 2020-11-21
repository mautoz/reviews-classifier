import pandas as pd
from helpers import format_aux

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







for linha in range(len(reviews)):
    print(f'{reviews["content"][linha]} - {find_app_name(reviews["app_Id"][linha])}')

print(len(reviews))

# print(reviews["userName"][0])

# print(len(reviews))


# for line in range(len(reviews)):
#     print(f'{reviews["content"][line]} - {reviews["score"][line]} - {reviews["app_Id"][line]}')



