import pandas as pd

# Google play store apps reviews 
# Header do CSV: reviewId, userName, userImage, content, score, thumbsUpCount
# reviewCreatedVersion, at, replyContent, repliedAt, app_Id
reviews = pd.read_csv('csv/google_play_store_apps_reviews.csv', nrows=6)

for linha in reviews:
    print(linha)

# print(reviews["userName"][0])

# print(len(reviews))


# for line in range(len(reviews)):
#     print(f'{reviews["content"][line]} - {reviews["score"][line]} - {reviews["app_Id"][line]}')

# Selecionando as colunas que interessam no CSV com o nome dos APPs e abrindo o CSV
colunas = ['appId', 'title']
apps_names = pd.read_csv('csv/apps.csv', usecols=colunas)
apps_names.dropna(inplace = True) 
names = apps_names["appId"].str.find('ginlemon.smartlauncher.extratool')

# Buscando o título do APP pelo seu ID
names = apps_names.loc[apps_names["appId"] == 'ginlemon.smartlauncher.extratool']
nome = names.applymap(str).iloc[0,1]
print(f'= {nome}')