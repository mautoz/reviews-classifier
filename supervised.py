import pandas as pd 
import os
import re

from helpers import db_aux

db_credentials = {
    'host' : os.getenv('POSTGRES_HOST'),
    'dbname' : os.getenv('POSTGRES_DATABASE'),
    'user' : os.getenv('POSTGRES_USER'),
    'password' : os.getenv('POSTGRES_PASSWORD'),
    'port' : os.getenv('POSTGRES_PORT') 
}


# Conecta no bd e busca todos os reviews que foram avaliados por humanos
with db_aux.connect_db(db_credentials) as conn:
    reviews = pd.DataFrame(db_aux.fetch_reviews(conn))
    reviews.rename(columns={0: "id", 1: "reviews_raw", 2: "a11y"}, inplace=True)
    print(reviews)

# Impressão do número de reviews por classificação humana
# print("+++")
# reviews.a11y.replace([0,1], ["not_a11y", "a11y"], inplace=True)
# print(reviews.a11y.value_counts())
# print("+++")
from sklearn.feature_extraction.text import CountVectorizer

vetorizar = CountVectorizer(lowercase=False, max_features=50)
bag_of_words = vetorizar.fit_transform(reviews["reviews_raw"])
print(bag_of_words.shape)

from sklearn.model_selection import train_test_split

treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words,
                                                              reviews["a11y"],
                                                              random_state = 42)

print("=====================================")
print(treino)
print("=====================================")
print(teste)
print("=====================================")
print(classe_treino)
print("=====================================") 
print(classe_teste)  



from sklearn.linear_model import LogisticRegression

regressao_logistica = LogisticRegression()
regressao_logistica.fit(treino, classe_treino)
acuracia = regressao_logistica.score(teste, classe_teste)

print(acuracia)


def classificar_texto(texto, coluna_texto, coluna_classificacao):
    vetorizar = CountVectorizer(lowercase=False, max_features=50)
    bag_of_words = vetorizar.fit_transform(texto[coluna_texto])
    treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words,
                                                              texto[coluna_classificacao],
                                                              random_state = 42)

    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(treino, classe_treino)
    return regressao_logistica.score(teste, classe_teste)

print("---")
print(classificar_texto(reviews, "reviews_raw", "a11y"))
print("---")

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def word_cloud_a11y(texto, coluna_texto):
    # Retorna as linhas que são de acessibilidade
    a11y_text = texto.query("a11y == 1")
    todas_palavras = ' '.join([texto for texto in a11y_text[coluna_texto]])

    nuvem_palavra = WordCloud(
        width=800, 
        height=600,
        max_font_size=110,
        collocations=False).generate(todas_palavras)

    plt.figure(figsize=(10,7))
    plt.imshow(nuvem_palavra, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("cloud.png")

word_cloud_a11y(reviews, "reviews_raw")


import nltk 
from nltk import tokenize

nltk.download("all")

a11y_text = texto.query("a11y == 1")
todas_palavras = ' '.join([texto for texto in a11y_text[reviews_raw]])

token_espaco = tokenize.WhitespaceTokenizer()
token_frase = token_espaco.tokenize(todas_palavras)
frequencia = nltk.FreqDist(token_frase)

print(frequencia)

# Bag of words / Vetorizar
# Visualizando no WordCloud
# Tokenização
# Tabela com as frequencias do token
# Stop Words
# Stop word de pontuações
# Normalização: retirar acentos
# Deixar o radical