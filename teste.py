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
print("---")


functions_aux.word_cloud_a11y(reviews, "reviews_raw")

import nltk 
from nltk import tokenize

# nltk.download("all")

a11y_text = reviews.query("a11y == 1")
todas_palavras = ' '.join([texto for texto in a11y_text["reviews_raw"]])

token_espaco = tokenize.WhitespaceTokenizer()
token_frase = token_espaco.tokenize(todas_palavras)
frequencia = nltk.FreqDist(token_frase)

df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()),
                              "Frequencia": list(frequencia.values())})

# print(df_frequencia.nlargest(columns = "Frequencia", n = 20))


# Desenha o gráfico de barras
functions_aux.word_frequency(reviews, "reviews_raw", "token")

from string import punctuation

# Vetor com pontuaçãoes como !, #, &...
def punctuation_array():
    marks = []
    for symbol in punctuation:
        marks.append(symbol)
    return marks


def stopwords_no_accent():  
    irrelevant_words = nltk.corpus.stopwords.words("portuguese")
    irrelevant_words_no_accent = [unidecode.unidecode(word) for word in irrelevant_words]
    return irrelevant_words_no_accent


# Para reviews em inglês, passar "english" como language. 
# Caso seja português, coloque "portuguese" no parâmetro.
def stop_words_format(texto, coluna_texto, language):
    irrelevant_words = nltk.corpus.stopwords.words(language)

    if language == "portuguese":
        irrelevant_words_no_accent = stopwords_no_accent()
        irrelevant_words = irrelevant_words + irrelevant_words_no_accent
    
    print(irrelevant_words)

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


def remove_punctuation(texto, coluna_texto):
    marks = punctuation_array()
    token_pontuacao = tokenize.WordPunctTokenizer()

    frase_processada = []
    for review in texto[coluna_texto]:
        nova_frase = []
        # phrase_no_dots = re.sub("[\.]+", "", review)
        palavras_texto = token_pontuacao.tokenize(review)
        for palavra in palavras_texto:
            if palavra not in marks:
                nova_frase.append(palavra)
        frase_processada.append(' '.join(nova_frase))
    return frase_processada


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
print(reviews.head())
functions_aux.word_frequency(reviews, "no_emojis", "no_emojis")
# Acurácia após retirar emojis
print("Acurácia sem emojis: ")
print(classificar_texto(reviews, "no_emojis", "a11y"))
print("-+-")

reviews["stop_words"] = stop_words_format(reviews, "no_emojis", "english")
# reviews["stop_words"] = stop_words_format(reviews, "reviews_raw", "portuguese")

print(reviews.head())

functions_aux.word_frequency(reviews, "stop_words", "stop_words")

# Acurácia após retirar as stopwords
print("Acurácia Stop Words: ")
print(classificar_texto(reviews, "stop_words", "a11y"))
print("---")

reviews["stop_words_punctuation"] = remove_punctuation(reviews, "stop_words")

# Acurácia após retirar as stopwords
print("Acurácia sem pontuação: ")
print(classificar_texto(reviews, "stop_words_punctuation", "a11y"))
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