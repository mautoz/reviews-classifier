import unidecode
import seaborn as sns
from nltk import tokenize
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud


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
    plt.savefig("./img/cloud.png")


def word_frequency(reviews, coluna_texto, fase):
    a11y_text = reviews.query("a11y == 1")
    todas_palavras = ' '.join([texto for texto in a11y_text[coluna_texto]])

    token_espaco = tokenize.WhitespaceTokenizer()
    token_frase = token_espaco.tokenize(todas_palavras)
    frequencia = nltk.FreqDist(token_frase)

    df_frequencia = pd.DataFrame({"Palavras": list(frequencia.keys()),
                                  "Frequencia": list(frequencia.values())})

    plt.figure(figsize=(12,8))
    ax = sns.barplot(
        data = df_frequencia.nlargest(columns="Frequencia", n=20), 
        x="Palavras", 
        y="Frequencia",
        color="blue")

    ax.set(ylabel="Contagem")

    nome = f'./img/frequencia_{fase}.png'
    plt.savefig(nome)





# Esta função foi retirada integralmente do site:
# https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
# Ela é útil pois muitos dos reviews contém emojis
import pandas as pd
import re

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


# Formatar palavra/expresão para inserção no banco de dados. Assim facilitamos
# na hora de realizar comparações de strings. Ideal para PT-BR, mas pode ser
# aplicado em outros casos de línguas.
# Exemplo: 
# input: BOtão difÍcil de LoCaliZar
# output: botao dificil de localizar
def formatar_texto(palavra):
    return unidecode.unidecode(palavra).lower()

def format_string(string_raw):
    string_treatment_1 = remove_emoji(string_raw)
    print(string_treatment_1)
    string_treatment_2 = formatar_texto(string_treatment_1)
    print(string_treatment_2)
    return string_treatment_2


string = "Botão difícil de Localizar So much it's so nice to play \U0001F5FF \U0001F6FF \U0001F1FF \U000027B0 \U0001F251"

print(format_string(string))