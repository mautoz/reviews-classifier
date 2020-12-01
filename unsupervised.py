import pandas as pd 
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

a11ys = pd.get_dummies(reviews["a11y"])
# print("+++++++++++++++++++")
a11ys.rename(columns={0: "not_a11y", 1: "a11y"}, inplace=True)
# print(a11ys)
# print("+++++++++++++++++++")


scaler = StandardScaler()
colunas_escaladas = scaler.fit_transform(a11ys)

modelo = KMeans(n_clusters=2)
modelo.fit(colunas_escaladas)

# print("Grupos {}".format(modelo.labels_))

# print(modelo.cluster_centers_)

# for item in modelo.labels_:
#     print(item)

# print(pd.DataFrame(modelo.cluster_centers_, 
#             columns=a11ys.columns))

def kmeans(num_clusters, colunas_escaladas):
    modelo = KMeans(n_clusters=num_clusters)
    modelo.fit(colunas_escaladas)

    return [num_clusters, modelo.inertia_]

resultado = [kmeans(numero, colunas_escaladas) for numero in range(1, 10)]

resultado = pd.DataFrame(resultado, 
            columns=['grupos', 'inertia'])

plot = resultado.inertia.plot(xticks=resultado.grupos)
fig = plot.get_figure()
fig.savefig("cotovelo.png")

print("=====================================")

cotovelo = KMeans(n_clusters=1)
cotovelo.fit(colunas_escaladas)

print("Grupos {}".format(modelo.labels_))

print(modelo.cluster_centers_)

print(type(modelo.labels_))

lista_modelo = modelo.labels_.tolist()
lista_modelo_2 = cotovelo.labels_.tolist()
lista_a11y = a11ys["a11y"].tolist()

i = 0
k = 0
for j in range(len(lista_modelo)):
    if lista_modelo[j] != lista_a11y[j]:
        i = i + 1
    if lista_modelo_2[j] != lista_a11y[j]:
        k = k + 1

print(i)
print(k)