from nltk.tokenize.regexp import WhitespaceTokenizer
import pandas as pd
import nltk

# Utilizar na primeira vez, depois deixar comentado
# nltk.download("all")

# Google play store apps reviews 
# Header do CSV: reviewId, userName, userImage, content, score, thumbsUpCount
# reviewCreatedVersion, at, replyContent, repliedAt, app_Id
reviews = pd.read_csv('csv/google_play_store_apps_reviews.csv', nrows=6)

for linha in reviews:
    print(linha)

# print(reviews["userName"][0])

# print(len(reviews))

# print(f'Antes: {reviews["content"][line]} - {reviews["score"][line]} - {reviews["app_Id"][line]}')


# Utilizando o NLTK para remover palavras sem importância.
# Será feito para PT e EN
palavras_sem_importancia = nltk.corpus.stopwords.words('english')
print(palavras_sem_importancia)

tk = WhitespaceTokenizer()
frases_processadas = []
for line in range(len(reviews)):
    nova_frase = []
    frase_tokenizada = tk.tokenize(reviews["content"][line])
    for palavra in frase_tokenizada:
        if palavra not in palavras_sem_importancia:
            nova_frase.append(palavra)
    frases_processadas.append(' '.join(nova_frase))
    print(f'Antes: {reviews["content"][line]}')
    print(f"Depois: {' '.join(nova_frase)}")


