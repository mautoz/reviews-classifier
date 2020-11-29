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


# reviews["not_a11y"] = reviews["a11y"].map({0: 1, 1: 0})

# a11ys = [reviews["not_a11y"], reviews["a11y"]]

a11ys = pd.get_dummies(reviews["a11y"])
print("+++++++++++++++++++")
a11ys.rename(columns={0: "not_a11y", 1: "a11y"}, inplace=True)
print(a11ys)
print("+++++++++++++++++++")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
colunas_escaladas = scaler.fit_transform(a11ys)
# for linha in colunas_escaladas:
#     print(linha)


from sklearn.cluster import KMeans

modelo = KMeans(n_clusters=2)

modelo.fit(colunas_escaladas)

# print("Grupos {}".format(modelo.labels_))

# print(modelo.cluster_centers_)

# for item in modelo.labels_:
#     print(item)

print(pd.DataFrame(modelo.cluster_centers_, 
            columns=a11ys.columns))

def kmeans(num_clusters, colunas_escaladas):
    modelo = KMeans(n_clusters=num_clusters)
    modelo.fit(colunas_escaladas)

    return [num_clusters, modelo.inertia_]

# resultado = [kmeans(numero, colunas_escaladas) for numero in range(1, 10)]

# resultado = pd.DataFrame(resultado, 
#             columns=['grupos', 'inertia'])



# plot = resultado.inertia.plot(xticks=resultado.grupos)
# fig = plot.get_figure()
# fig.savefig("cotovelo.png")

print("=====================================")

modelo2 = KMeans(n_clusters=1)
modelo2.fit(colunas_escaladas)

print("Grupos {}".format(modelo.labels_))

print(modelo.cluster_centers_)

