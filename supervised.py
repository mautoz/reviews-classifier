import pandas as pd 
import os
import re
import unidecode

from helpers import db_aux, functions_aux


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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def classificar_texto(texto, coluna_texto, coluna_classificacao):
    vetorizar = CountVectorizer(lowercase=False, max_features=50)
    bag_of_words = vetorizar.fit_transform(texto[coluna_texto])
    treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words,
                                                              texto[coluna_classificacao],
                                                              random_state = 42)

    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(treino, classe_treino)
    return regressao_logistica.score(teste, classe_teste)

# Resultado da acurácia sem tratamento algum
print("Acurácia sem tratamento: ")
print(classificar_texto(reviews, "reviews_raw", "a11y"))
# Desenha a nuvem de 'reviews_raw'
functions_aux.word_cloud_a11y(reviews, "reviews_raw")
# Desenha o gráfico de barras
functions_aux.word_frequency(reviews, "reviews_raw", "token")
print("---")

import nltk 
from nltk import tokenize

# A linha abaixo deve ser executada na primeira vez, após o download pode comentar
# para não ficar executando novamente.
# nltk.download("all")

from string import punctuation

# Vetor com pontuaçãoes como !, #, &...
def punctuation_array():
    marks = []
    for symbol in punctuation:
        marks.append(symbol)
    return marks


# Para reviews em português é possível que existam palavras como 'nao', 'sao'
# que eram para ter acentos, mas não tem! Então precisamos remover os acentos
# das stop words para que elas sejam removidas dos reviews.
def stopwords_no_accent():  
    irrelevant_words = nltk.corpus.stopwords.words("portuguese")
    irrelevant_words_no_accent = [unidecode.unidecode(word) for word in irrelevant_words]
    return irrelevant_words_no_accent


# Para reviews em inglês, passar "english" como language. 
# Caso seja português, coloque "portuguese" no parâmetro.
def stop_words_format(texto, coluna_texto, language):
    token_espaco = tokenize.WhitespaceTokenizer()
    irrelevant_words = nltk.corpus.stopwords.words(language)

    if language == "portuguese":
        irrelevant_words_no_accent = stopwords_no_accent()
        irrelevant_words = irrelevant_words + irrelevant_words_no_accent
    
    frase_processada = []
    for review in texto[coluna_texto]:
        nova_frase = []
        review_lower = review.lower()
        palavras_texto = token_espaco.tokenize(review_lower)
        for palavra in palavras_texto:
            if palavra not in irrelevant_words:
                nova_frase.append(palavra)
        frase_processada.append(' '.join(nova_frase))
    return frase_processada


# Remover pontuação dos reviews
def remove_punctuation(texto, coluna_texto):
    marks = punctuation_array()
    token_pontuacao = tokenize.WordPunctTokenizer()

    frase_processada = []
    for review in texto[coluna_texto]:
        nova_frase = []
        # Existem reviews com excesso de reticências! É tão grande a frequência
        #  que estavam entrando no top 20 das mais frequentes dos gráficos.
        phrase_no_dots = re.sub("[\.]+", "", review)
        palavras_texto = token_pontuacao.tokenize(phrase_no_dots)
        for palavra in palavras_texto:
            if palavra not in marks:
                nova_frase.append(palavra)
        frase_processada.append(' '.join(nova_frase))
    return frase_processada


# Pegar a raiz das palavras! Por exemplo: 'like' e 'likes' tem 
# o mesmo sentido na classificação, assim como 'fix' ou 'fixing'.
def stemming_word(texto, coluna_texto):
    stemmer = nltk.RSLPStemmer()
    token_pontuacao = tokenize.WordPunctTokenizer()

    frase_processada = []
    for review in texto[coluna_texto]:
        nova_frase = []
        palavras_texto = token_pontuacao.tokenize(review)
        for palavra in palavras_texto:
            nova_frase.append(stemmer.stem(palavra))
        frase_processada.append(' '.join(nova_frase))
    return frase_processada

reviews["no_emojis"] = functions_aux.format_string(reviews, "reviews_raw")
# Acurácia após retirar emojis
print("Acurácia sem emojis: ")
print(classificar_texto(reviews, "no_emojis", "a11y"))
# Desenha o gráfico de frequência sem emojis
functions_aux.word_frequency(reviews, "no_emojis", "no_emojis")
print("-+-")

reviews["stop_words"] = stop_words_format(reviews, "no_emojis", "english")
# reviews["stop_words"] = stop_words_format(reviews, "reviews_raw", "portuguese")
# Acurácia após retirar as stopwords
print("Acurácia Stop Words: ")
print(classificar_texto(reviews, "stop_words", "a11y"))
# Desenha o gráfico de frequências após retirar as stop_wprds
functions_aux.word_frequency(reviews, "stop_words", "stop_words")
print("---")

reviews["stop_words_punctuation"] = remove_punctuation(reviews, "stop_words")
# Acurácia após retirar as stopwords
print("Acurácia sem pontuação: ")
print(classificar_texto(reviews, "stop_words_punctuation", "a11y"))
functions_aux.word_frequency(reviews, "stop_words_punctuation", "stop_words_punctuation")
print("---")

reviews["stemming"] = stemming_word(reviews, "stop_words_punctuation")
# Acurácia após retirar as stopwords
functions_aux.word_frequency(reviews, "stemming", "stemming")


from sklearn.feature_extraction.text import TfidfVectorizer 

def classificar_texto_tfidf(texto, coluna_texto, coluna_classificacao):
    tfidf = TfidfVectorizer(lowercase=False, max_features=50)
    bag_of_words = tfidf.fit_transform(texto[coluna_texto])
    treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words,
                                                              texto[coluna_classificacao],
                                                              random_state = 42)

    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(treino, classe_treino)
    return regressao_logistica.score(teste, classe_teste)


def classificar_texto_ngrams(texto, coluna_texto, coluna_classificacao):
    tfidf = TfidfVectorizer(lowercase=False, ngram_range= (1,2))
    vector_tfidf = tfidf.fit_transform(texto[coluna_texto])
    treino, teste, classe_treino, classe_teste = train_test_split(vector_tfidf,
                                                              texto[coluna_classificacao],
                                                              random_state = 42)

    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(treino, classe_treino)

    # Colocar isso nos demais
    pesos = pd.DataFrame(
        regressao_logistica.coef_[0].T,
        index = tfidf.get_feature_names()
    )

    return regressao_logistica.score(teste, classe_teste), pesos

print("===")
print(classificar_texto(reviews, "stemming", "a11y"))
print("===")
print(classificar_texto_tfidf(reviews, "stemming", "a11y"))
print("===")
acuracia, pesos = classificar_texto_ngrams(reviews, "stemming", "a11y")
print(acuracia)
print(pesos.nlargest(10,0))